#!/usr/bin/env python3
"""Generate deterministic TFLite FC models for Metal precision benchmarks.

Generate the TFLite Python bindings with flatc and put both those bindings and
the vendored FlatBuffers Python runtime on PYTHONPATH before running this tool.
"""

import argparse
from pathlib import Path

import numpy as np

try:
    import flatbuffers
    from tflite.ActivationFunctionType import ActivationFunctionType
    from tflite.Buffer import BufferT
    from tflite.BuiltinOperator import BuiltinOperator
    from tflite.BuiltinOptions import BuiltinOptions
    from tflite.CustomOptionsFormat import CustomOptionsFormat
    from tflite.DequantizeOptions import DequantizeOptionsT
    from tflite.FullyConnectedOptions import FullyConnectedOptionsT
    from tflite.FullyConnectedOptionsWeightsFormat import (
        FullyConnectedOptionsWeightsFormat,
    )
    from tflite.Model import ModelT
    from tflite.Operator import OperatorT
    from tflite.OperatorCode import OperatorCodeT
    from tflite.QuantizationParameters import QuantizationParametersT
    from tflite.SubGraph import SubGraphT
    from tflite.Tensor import TensorT
    from tflite.TensorType import TensorType
except ImportError as error:
    raise SystemExit(
        "generate TFLite Python bindings and add them plus the FlatBuffers "
        "Python runtime to PYTHONPATH"
    ) from error


def deterministic_weights(outputs: int, inputs: int) -> np.ndarray:
    indices = np.arange(outputs * inputs, dtype=np.uint64)
    values = (
        indices * np.uint64(1664525) + np.uint64(0x2468ACE1)
    ) & np.uint64(0xFFFFFFFF)
    normalized = (
        (values >> np.uint64(8)).astype(np.float32) / np.float32(8388607.5)
        - 1.0
    )
    return (normalized * 0.03).reshape(outputs, inputs)


def tensor(shape, dtype, buffer, name, quantization=None):
    return TensorT(
        shape=np.asarray(shape, dtype=np.int32),
        type=dtype,
        buffer=buffer,
        name=name,
        quantization=quantization,
        isVariable=False,
        hasRank=True,
    )


def make_model(
    path: Path, rows: int, inputs: int, outputs: int, precision: str
) -> None:
    weights = deterministic_weights(outputs, inputs)
    quantization = None
    if precision == "fp32":
        weight_type = TensorType.FLOAT32
        weight_data = weights.astype("<f4", copy=False).view(np.uint8).reshape(-1)
    elif precision == "fp16":
        weight_type = TensorType.FLOAT16
        weight_data = weights.astype("<f2").view(np.uint8).reshape(-1)
    elif precision == "int8":
        weight_type = TensorType.INT8
        scales = np.max(np.abs(weights), axis=1) / np.float32(127.0)
        scales[scales == 0] = np.float32(1.0)
        quantized = np.clip(
            np.rint(weights / scales[:, None]), -127, 127
        ).astype(np.int8)
        weight_data = quantized.view(np.uint8).reshape(-1)
        quantization = QuantizationParametersT(
            scale=scales.astype("<f4"),
            zeroPoint=np.zeros(outputs, dtype="<i8"),
            quantizedDimension=0,
        )
    else:
        raise ValueError(f"unsupported precision: {precision}")

    tensors = [
        tensor([rows, inputs], TensorType.FLOAT32, 0, "input"),
        tensor([outputs, inputs], weight_type, 1, "weight", quantization),
    ]
    options = FullyConnectedOptionsT(
        fusedActivationFunction=ActivationFunctionType.NONE,
        weightsFormat=FullyConnectedOptionsWeightsFormat.DEFAULT,
        keepNumDims=True,
        asymmetricQuantizeInputs=False,
        quantizedBiasType=TensorType.FLOAT32,
    )
    operators = []
    opcodes = []
    if precision == "fp16":
        tensors.append(
            tensor(
                [outputs, inputs], TensorType.FLOAT32, 0, "dequantized_weight"
            )
        )
        tensors.append(tensor([rows, outputs], TensorType.FLOAT32, 0, "output"))
        opcodes.append(
            OperatorCodeT(
                deprecatedBuiltinCode=BuiltinOperator.DEQUANTIZE,
                builtinCode=BuiltinOperator.DEQUANTIZE,
                version=2,
            )
        )
        operators.append(
            OperatorT(
                opcodeIndex=0,
                inputs=np.asarray([1], dtype=np.int32),
                outputs=np.asarray([2], dtype=np.int32),
                builtinOptionsType=BuiltinOptions.DequantizeOptions,
                builtinOptions=DequantizeOptionsT(),
                customOptionsFormat=CustomOptionsFormat.FLEXBUFFERS,
            )
        )
        weight_index = 2
        output_index = 3
    else:
        tensors.append(tensor([rows, outputs], TensorType.FLOAT32, 0, "output"))
        weight_index = 1
        output_index = 2

    opcodes.append(
        OperatorCodeT(
            deprecatedBuiltinCode=BuiltinOperator.FULLY_CONNECTED,
            builtinCode=BuiltinOperator.FULLY_CONNECTED,
            version=5,
        )
    )
    operators.append(
        OperatorT(
            opcodeIndex=len(opcodes) - 1,
            inputs=np.asarray([0, weight_index, -1], dtype=np.int32),
            outputs=np.asarray([output_index], dtype=np.int32),
            builtinOptionsType=BuiltinOptions.FullyConnectedOptions,
            builtinOptions=options,
            customOptionsFormat=CustomOptionsFormat.FLEXBUFFERS,
        )
    )
    graph = SubGraphT(
        tensors=tensors,
        inputs=np.asarray([0], dtype=np.int32),
        outputs=np.asarray([output_index], dtype=np.int32),
        operators=operators,
        name="main",
    )
    model = ModelT(
        version=3,
        operatorCodes=opcodes,
        subgraphs=[graph],
        description=f"Kidi Metal {precision} FC benchmark",
        buffers=[BufferT(), BufferT(data=weight_data)],
    )
    builder = flatbuffers.Builder(max(1024, weight_data.size + 4096))
    root = model.Pack(builder)
    builder.Finish(root, file_identifier=b"TFL3")
    path.write_bytes(builder.Output())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--inputs", type=int, default=768)
    parser.add_argument("--outputs", type=int, default=64000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for precision in ("fp32", "fp16", "int8"):
        path = args.output_dir / f"fc_{precision}.tflite"
        make_model(path, args.rows, args.inputs, args.outputs, precision)
        print(f"{precision}={path} bytes={path.stat().st_size}")


if __name__ == "__main__":
    main()
