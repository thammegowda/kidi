// Images attached to the next message: picked, pasted or dropped, shown as removable thumbnails.
import {blobToDataUrl} from './liteapp/media/base64.mjs';
import {prepareImage} from './liteapp/media/images.mjs';

const MAX_IMAGES = 8;
const strip = document.getElementById('attachments');
const input = document.getElementById('attach-input');
let images = [];
let onError = () => {};

function render() {
  strip.replaceChildren(...images.map((url, index) => {
    const item = document.createElement('div');
    item.className = 'attachment';
    const thumbnail = document.createElement('img');
    thumbnail.src = url;
    thumbnail.alt = `Attached image ${index + 1}`;
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.textContent = '×';
    remove.title = 'Remove';
    remove.addEventListener('click', () => {
      images.splice(index, 1);
      render();
    });
    item.append(thumbnail, remove);
    return item;
  }));
  strip.hidden = !images.length;
}

export async function addImages(files) {
  for (const file of files) {
    if (!file.type.startsWith('image/')) continue;
    if (images.length >= MAX_IMAGES) return onError(`Attach at most ${MAX_IMAGES} images.`);
    try {
      images.push(await blobToDataUrl(await prepareImage(file)));
    } catch {
      onError(`Cannot read ${file.name || 'the image'}.`);
    }
    render();
  }
}

// Returns the attached images and clears the strip.
export function takeImages() {
  const taken = images;
  images = [];
  render();
  return taken;
}

export function hasImages() { return images.length > 0; }

export function setupAttachments(errorHandler) {
  onError = errorHandler;
  document.getElementById('attach').addEventListener('click', () => input.click());
  input.addEventListener('change', () => {
    addImages([...input.files]);
    input.value = '';
  });
  const composer = document.getElementById('composer');
  composer.addEventListener('paste', event => {
    const files = [...(event.clipboardData?.files ?? [])].filter(file => file.type.startsWith('image/'));
    if (!files.length) return;
    event.preventDefault();
    addImages(files);
  });
  composer.addEventListener('dragover', event => {
    if (![...event.dataTransfer.types].includes('Files')) return;
    event.preventDefault();
    composer.classList.add('dropping');
  });
  composer.addEventListener('dragleave', () => composer.classList.remove('dropping'));
  composer.addEventListener('drop', event => {
    composer.classList.remove('dropping');
    if (!event.dataTransfer.files.length) return;
    event.preventDefault();
    addImages([...event.dataTransfer.files]);
  });
}
