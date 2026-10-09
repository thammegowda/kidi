#include "kidi_camera.h"

#include <fcntl.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "accessory_protocol.h"
#include "esp_check.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "example_video_common.h"
#include "linux/videodev2.h"

#define KIDI_CAMERA_BUFFER_COUNT 2U
#define KIDI_CAMERA_WARMUP_FRAMES 2U
#define KIDI_CAMERA_JPEG_QUALITY 80U

static const char* TAG = "kidi_camera";

typedef struct {
    int descriptor;
    uint8_t* buffers[KIDI_CAMERA_BUFFER_COUNT];
    size_t buffer_sizes[KIDI_CAMERA_BUFFER_COUNT];
    uint32_t width;
    uint32_t height;
    uint32_t pixel_format;
    example_encoder_handle_t encoder;
    uint8_t* encoded;
    uint32_t encoded_capacity;
    bool streaming;
    bool video_initialized;
} camera_session_t;

static void close_session(camera_session_t* session) {
    if (session->streaming) {
        int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        ioctl(session->descriptor, VIDIOC_STREAMOFF, &type);
    }
    if (session->encoded != NULL) {
        example_encoder_free_output_buffer(session->encoder, session->encoded);
    }
    if (session->encoder != NULL) {
        example_encoder_deinit(session->encoder);
    }
    for (size_t index = 0; index < KIDI_CAMERA_BUFFER_COUNT; ++index) {
        if (session->buffers[index] != NULL && session->buffers[index] != MAP_FAILED) {
            munmap(session->buffers[index], session->buffer_sizes[index]);
        }
    }
    if (session->descriptor >= 0) {
        close(session->descriptor);
    }
    if (session->video_initialized) {
        esp_err_t error = example_video_deinit();
        if (error != ESP_OK) {
            ESP_LOGE(TAG, "failed to deinitialize camera pipeline: %s", esp_err_to_name(error));
        }
    }
    *session = {};
    session->descriptor = -1;
}

static esp_err_t open_session(camera_session_t* session) {
    *session = {};
    session->descriptor = -1;
    ESP_RETURN_ON_ERROR(example_video_init(), TAG, "failed to initialize camera pipeline");
    session->video_initialized = true;

    session->descriptor = open(ESP_VIDEO_MIPI_CSI_DEVICE_NAME, O_RDWR);
    ESP_RETURN_ON_FALSE(session->descriptor >= 0, ESP_ERR_NOT_FOUND, TAG, "failed to open MIPI camera");

    v4l2_format format{};
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ESP_RETURN_ON_FALSE(ioctl(session->descriptor, VIDIOC_G_FMT, &format) == 0, ESP_FAIL, TAG,
                        "failed to read camera format");
    session->width = format.fmt.pix.width;
    session->height = format.fmt.pix.height;
    session->pixel_format = format.fmt.pix.pixelformat;

    v4l2_requestbuffers request{};
    request.count = KIDI_CAMERA_BUFFER_COUNT;
    request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    request.memory = V4L2_MEMORY_MMAP;
    ESP_RETURN_ON_FALSE(ioctl(session->descriptor, VIDIOC_REQBUFS, &request) == 0, ESP_FAIL, TAG,
                        "failed to request camera buffers");
    ESP_RETURN_ON_FALSE(request.count >= KIDI_CAMERA_BUFFER_COUNT, ESP_ERR_NO_MEM, TAG,
                        "camera returned only %u buffers", request.count);

    for (size_t index = 0; index < KIDI_CAMERA_BUFFER_COUNT; ++index) {
        v4l2_buffer buffer{};
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.index = index;
        ESP_RETURN_ON_FALSE(ioctl(session->descriptor, VIDIOC_QUERYBUF, &buffer) == 0, ESP_FAIL, TAG,
                            "failed to query camera buffer");
        session->buffer_sizes[index] = buffer.length;
        void* mapped =
            mmap(NULL, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, session->descriptor, buffer.m.offset);
        ESP_RETURN_ON_FALSE(mapped != MAP_FAILED, ESP_ERR_NO_MEM, TAG, "failed to map camera buffer");
        session->buffers[index] = static_cast<uint8_t*>(mapped);
        ESP_RETURN_ON_FALSE(ioctl(session->descriptor, VIDIOC_QBUF, &buffer) == 0, ESP_FAIL, TAG,
                            "failed to queue camera buffer");
    }

    if (session->pixel_format != V4L2_PIX_FMT_JPEG) {
        example_encoder_config_t encoder_config{};
        encoder_config.width = session->width;
        encoder_config.height = session->height;
        encoder_config.pixel_format = session->pixel_format;
        encoder_config.quality = KIDI_CAMERA_JPEG_QUALITY;
        ESP_RETURN_ON_ERROR(example_encoder_init(&encoder_config, &session->encoder), TAG,
                            "failed to initialize JPEG encoder");
        ESP_RETURN_ON_ERROR(
            example_encoder_alloc_output_buffer(session->encoder, &session->encoded, &session->encoded_capacity), TAG,
            "failed to allocate JPEG output");
    }

    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ESP_RETURN_ON_FALSE(ioctl(session->descriptor, VIDIOC_STREAMON, &type) == 0, ESP_FAIL, TAG,
                        "failed to start camera stream");
    session->streaming = true;
    ESP_LOGI(TAG, "camera ready: %" PRIu32 "x%" PRIu32 " format=" V4L2_FMT_STR, session->width, session->height,
             V4L2_FMT_STR_ARG(session->pixel_format));
    return ESP_OK;
}

static esp_err_t dequeue(camera_session_t* session, struct v4l2_buffer* buffer) {
    *buffer = {};
    buffer->type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer->memory = V4L2_MEMORY_MMAP;
    ESP_RETURN_ON_FALSE(ioctl(session->descriptor, VIDIOC_DQBUF, buffer) == 0, ESP_FAIL, TAG,
                        "failed to dequeue camera frame");
    if (!(buffer->flags & V4L2_BUF_FLAG_DONE) || buffer->index >= KIDI_CAMERA_BUFFER_COUNT) {
        ioctl(session->descriptor, VIDIOC_QBUF, buffer);
        return ESP_ERR_INVALID_RESPONSE;
    }
    return ESP_OK;
}

static esp_err_t discard_frame(camera_session_t* session) {
    struct v4l2_buffer buffer;
    ESP_RETURN_ON_ERROR(dequeue(session, &buffer), TAG, "failed to capture warmup frame");
    ESP_RETURN_ON_FALSE(ioctl(session->descriptor, VIDIOC_QBUF, &buffer) == 0, ESP_FAIL, TAG,
                        "failed to return warmup frame");
    return ESP_OK;
}

static esp_err_t copy_jpeg(camera_session_t* session, kidi_camera_photo_t* photo) {
    struct v4l2_buffer buffer;
    ESP_RETURN_ON_ERROR(dequeue(session, &buffer), TAG, "failed to capture photo");
    const uint8_t* jpeg = session->buffers[buffer.index];
    uint32_t jpeg_size = buffer.bytesused;
    esp_err_t error = ESP_OK;
    if (session->pixel_format != V4L2_PIX_FMT_JPEG) {
        error = example_encoder_process(session->encoder, session->buffers[buffer.index],
                                        session->buffer_sizes[buffer.index], session->encoded,
                                        session->encoded_capacity, &jpeg_size);
        jpeg = session->encoded;
    }
    if (error == ESP_OK && (jpeg_size < 4 || jpeg_size > KIDI_ACCESSORY_MAX_PHOTO_BYTES || jpeg[0] != 0xff ||
                            jpeg[1] != 0xd8 || jpeg[jpeg_size - 2] != 0xff || jpeg[jpeg_size - 1] != 0xd9)) {
        error = ESP_ERR_INVALID_RESPONSE;
    }
    if (error == ESP_OK) {
        photo->data = static_cast<uint8_t*>(heap_caps_malloc(jpeg_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
        if (photo->data == NULL) {
            error = ESP_ERR_NO_MEM;
        } else {
            memcpy(photo->data, jpeg, jpeg_size);
            photo->size = jpeg_size;
            photo->width = session->width;
            photo->height = session->height;
        }
    }
    if (ioctl(session->descriptor, VIDIOC_QBUF, &buffer) != 0 && error == ESP_OK) {
        error = ESP_FAIL;
    }
    return error;
}

esp_err_t kidi_camera_capture_photo(kidi_camera_photo_t* photo) {
    if (photo == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    *photo = {};
    camera_session_t session;
    esp_err_t error = open_session(&session);
    if (error == ESP_OK) {
        for (size_t index = 0; index < KIDI_CAMERA_WARMUP_FRAMES && error == ESP_OK; ++index) {
            error = discard_frame(&session);
        }
    }
    if (error == ESP_OK) {
        error = copy_jpeg(&session, photo);
    }
    close_session(&session);
    if (error != ESP_OK) {
        kidi_camera_release_photo(photo);
    }
    return error;
}

void kidi_camera_release_photo(kidi_camera_photo_t* photo) {
    if (photo == NULL) {
        return;
    }
    free(photo->data);
    *photo = {};
}
