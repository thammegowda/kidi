"""Check the matched Arm benchmark protocol without loading Gemma."""

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("arm_gemma_benchmark", ROOT / "benchmarks/gemma4/arm/run.py")
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


class ArmGemmaBenchmarkTest(unittest.TestCase):
    def protocol(self):
        return {"input_token_ids": [2, 3], "output_tokens": 4, "warmups": 1, "repeats": 3, "threads": 4}

    def records(self):
        records = [{"stage": "runtime", "device": "cpu:0", "threads": 4}]
        for index, speed in enumerate((1, 10, 20, 30)):
            records.append({"stage": "generation", "iteration": index, "warmup": index == 0,
                            "input_token_ids": [2, 3], "prompt_tokens": 2, "token_ids": [4, 5, 6, 7],
                            "output_tokens": 4, "decode_tokens": 3, "first_token_seconds": 0.01,
                            "decode_seconds": 3 / speed, "prefill_through_first_token_tps": 200,
                            "decode_interval_tps": speed, "peak_rss_kib": 2**20})
        return records

    def test_warm_summary(self):
        summary = BENCHMARK.summarize(self.records(), self.protocol())
        self.assertEqual(summary["decode_tps"], 20)
        self.assertEqual(summary["decode_range"], [10, 30])
        self.assertEqual(summary["ttft_ms"], 10)
        self.assertEqual(summary["peak_rss_gib"], 1)
        self.assertEqual(BENCHMARK.format_progress_time(0), "00:00")
        self.assertEqual(BENCHMARK.format_progress_time(65), "01:05")
        self.assertEqual(BENCHMARK.format_progress_time(3661), "1:01:01")

    def test_rejects_invalid_or_unmatched_measurements(self):
        for change in ("device", "threads", "missing", "warmup", "prompt", "count", "tokens", "reuse", "nan", "rate"):
            with self.subTest(change=change):
                rows = self.records()
                if change == "device":
                    rows[0]["device"] = "mps:0"
                elif change == "threads":
                    rows[0]["threads"] = 8
                elif change == "missing":
                    rows.pop()
                elif change == "warmup":
                    rows[-1]["warmup"] = True
                elif change == "prompt":
                    rows[-1]["input_token_ids"] = [99]
                elif change == "count":
                    rows[-1]["decode_tokens"] = 4
                elif change == "tokens":
                    rows[-1]["token_ids"][0] = 99
                elif change == "reuse":
                    rows[-1]["reused_prompt_tokens"] = 1
                elif change == "rate":
                    rows[-1]["decode_interval_tps"] = 100
                else:
                    rows[-1]["decode_interval_tps"] = float("nan")
                with self.assertRaises(ValueError):
                    BENCHMARK.summarize(rows, self.protocol())

    def test_independent_greedy_comparison_reports_different_outputs(self):
        row = BENCHMARK.summarize(self.records(), self.protocol())
        summaries = {"kidi": row, "arm": copy.deepcopy(row)}
        comparison = BENCHMARK.compare(summaries)
        self.assertEqual(comparison["speedup"], {"prefill_tps": 1, "decode_tps": 1})
        self.assertTrue(comparison["outputs"]["identical"])
        summaries["arm"]["token_ids"][0] = 99
        self.assertEqual(BENCHMARK.compare(summaries)["outputs"]["first_divergent_token"], 1)
        summaries["arm"]["token_ids"].pop()
        with self.assertRaisesRegex(ValueError, "output token budgets differ"):
            BENCHMARK.compare(summaries)

    def test_packing_and_embedding_layout_preserve_values(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("optional export dependencies are not installed")
        import torch
        weight = torch.tensor([[-7., -1., 0., 1., 7., -3.5], [0., 0., 0., 0., 0., 0.]])
        values, scales = BENCHMARK.pack_int4(weight, chunk_rows=1)
        self.assertTrue(torch.equal(scales, torch.tensor([[1.], [1.]])))
        self.assertEqual(BENCHMARK.unpack_int4(values).tolist(), [[-7, -1, 0, 1, 7, -4], [0, 0, 0, 0, 0, 0]])
        self.assertTrue(torch.equal(BENCHMARK.embedding_layout(BENCHMARK.embedding_layout(values)), values))
        below_half = torch.nextafter(torch.tensor(0.5), torch.tensor(0.)).item()
        values, _ = BENCHMARK.pack_int4(torch.tensor([[below_half, 7.]]))
        self.assertEqual(BENCHMARK.unpack_int4(values).tolist(), [[0, 7]])
        grouped = torch.tensor([[-7., 3.5, 14., -7.]])
        grouped_values, grouped_scales = BENCHMARK.pack_int4(grouped, group_size=2, chunk_rows=1)
        self.assertTrue(torch.equal(grouped_scales, torch.tensor([[1., 2.]])))
        self.assertEqual(BENCHMARK.unpack_int4(grouped_values).tolist(), [[-7, 4, 7, -4]])
        int8_values, int8_scales = BENCHMARK.pack_int8(torch.tensor([[-127., 0., 63.5, 127.]]), chunk_rows=1)
        self.assertTrue(torch.equal(int8_scales, torch.tensor([[1.]])))
        self.assertEqual(int8_values.view(torch.int8).tolist(), [[-127, 0, 64, 127]])
        asym_values, asym_scales, asym_zeros = BENCHMARK.pack_int4_asym_clip(
            torch.tensor([[-0.2, 0.0, 0.5, 1.0]]), group_size=2, chunk_rows=1, asymmetric=True
        )
        self.assertEqual(asym_values.shape, (1, 2))
        self.assertEqual(asym_scales.shape, (1, 2))
        self.assertEqual(asym_zeros.shape, (1, 2))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            BENCHMARK.pack_int4(torch.tensor([[float("nan"), 0.]]))

    def test_consolidation_preserves_integer_values_and_scales(self):
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("safetensors") is None:
            self.skipTest("optional export dependencies are not installed")
        import torch
        from safetensors.torch import load_file, save_file

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            values = torch.tensor([[0x19, 0x07]], dtype=torch.uint8)
            scales = torch.tensor([[0.125]], dtype=torch.float32)
            save_file({"weight": values}, str(directory / "model-0000.safetensors"))
            save_file({"weight_scale": scales}, str(directory / "model-0001.safetensors"))
            BENCHMARK.consolidate_weights(directory, ["model-0000.safetensors", "model-0001.safetensors"])
            merged = load_file(str(directory / "model.safetensors"))
            self.assertTrue(torch.equal(merged["weight"], values))
            self.assertTrue(torch.equal(merged["weight_scale"], scales))
            self.assertFalse((directory / "model-0000.safetensors").exists())
            with self.assertRaisesRegex(ValueError, "no existing"):
                BENCHMARK.consolidate_weights(directory, ["model-0000.safetensors"])

    def test_cached_run_reuse_and_invalidation(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            protocol = self.protocol()
            records = self.records()
            records.insert(1, {"stage": "load", "native_qat": True})
            run = {
                "command": ["bench", "model"],
                "binary_sha256": "abc123",
                "records": records,
                "summary": BENCHMARK.summarize(records, protocol),
            }
            BENCHMARK.save_cached_run(work, "kidi", run, protocol)
            loaded = BENCHMARK.load_cached_run(work, "kidi", ["bench", "model"], "abc123", protocol)
            self.assertIsNotNone(loaded)
            self.assertTrue(loaded["cached"])
            self.assertEqual(loaded["summary"]["decode_tps"], 20)
            self.assertIsNone(BENCHMARK.load_cached_run(work, "kidi", ["bench", "other"], "abc123", protocol))
            self.assertIsNone(BENCHMARK.load_cached_run(work, "kidi", ["bench", "model"], "changed", protocol))
            changed_protocol = {**protocol, "threads": 8}
            self.assertIsNone(BENCHMARK.load_cached_run(work, "kidi", ["bench", "model"], "abc123", changed_protocol))

    def test_eval_cleaning_and_cache_jsonl(self):
        prompt = BENCHMARK.serialize_translation_prompt("uk", "Тест")
        self.assertTrue(prompt.startswith("<bos><|turn>system\nYou are a professional translator."))
        self.assertIn("<turn|>\n<|turn>user\nTranslate the following Ukrainian text fully into English.", prompt)
        self.assertTrue(prompt.endswith("Тест<turn|>\n<|turn>model\n"))
        self.assertEqual(
            BENCHMARK.clean_translation_text("<|channel>thought\n thinking <channel|>\nHello world\n<turn|>"),
            "Hello world",
        )
        if importlib.util.find_spec("sacrebleu") is None:
            self.skipTest("optional sacrebleu dependency is not installed")

        class FakeTokenizer:
            def decode(self, ids, skip_special_tokens=True):
                return "Photo: t.me/qr_tjm" if ids == [10, 11] else "Great backpack"

        eval_protocol = {
            "mode": "eval",
            "task": "wmt23",
            "langs": ["ja", "uk"],
            "seed": 42,
            "items": 2,
            "prompts": ["p0", "p1"],
            "input_token_ids_list": [[1, 2], [3, 4]],
            "sources": ["s0", "s1"],
            "references": ["Photo: t.me/qr_tjm", "Great backpack"],
            "item_langs": ["ja", "uk"],
            "max_output_tokens": 32,
            "context_tokens": 1024,
            "warmups": 1,
            "threads": 4,
        }
        records = [{
            "stage": "eval",
            "items": 2,
            "eval_seconds": 0.5,
            "prompt_tokens": 20,
            "output_tokens": 4,
            "token_ids": [[10, 11, 106], [12, 13, 1]],
        }]
        summary = BENCHMARK.summarize_eval(records, eval_protocol, FakeTokenizer())
        self.assertAlmostEqual(summary["chrf"], 100.0, places=3)
        self.assertEqual(summary["output_tokens"], 4)
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            eval_run = {"command": ["bench"], "binary_sha256": "abc", "records": records, "summary": summary}
            BENCHMARK.save_cached_eval(work, "kidi", eval_run, eval_protocol)
            output_jsonl = (work / "results" / "kidi_eval.output.jsonl").read_text().splitlines()
            self.assertEqual(len(output_jsonl), 2)
            loaded = BENCHMARK.load_cached_eval(work, "kidi", ["bench"], "abc", eval_protocol, FakeTokenizer())
            self.assertIsNotNone(loaded)
            self.assertTrue(loaded["cached"])
            self.assertAlmostEqual(loaded["summary"]["chrf"], 100.0, places=3)
            changed_prompt = copy.deepcopy(eval_protocol)
            changed_prompt["input_token_ids_list"][0] = [99]
            self.assertEqual(BENCHMARK.load_partial_eval_items(work, "kidi", changed_prompt, FakeTokenizer()), {})

    def test_results_tsv_and_readme_tables(self):
        rows = BENCHMARK.load_results_tsv()
        self.assertEqual(len(rows), 14)
        self.assertEqual({r["tier"] for r in rows}, {"reference", "q8", "q4", "reference_only"})
        for row in rows:
            for key in ("model_gib", "prefill_tps", "decode_tps", "ttft_ms", "peak_rss_gib", "chrf_ref"):
                self.assertGreater(float(row[key]), 0)
        rendered = BENCHMARK.render_markdown_tables()
        readme = (BENCHMARK.HERE / "README.md").read_text(encoding="utf-8")
        self.assertIn(rendered, readme)
        svg = BENCHMARK.render_pareto_svg()
        self.assertEqual(svg.count(">Kidi Q8+HNSW</text>"), 1)
        self.assertEqual(svg.count(">Kidi Q4_g32+HNSW</text>"), 1)
        self.assertEqual(svg.count(">Kidi Q4 (g=128)+HNSW</text>"), 1)
        self.assertNotIn("LiteRT", svg)
        svg_path = ROOT / "docs/gemma4-arm-pareto.svg"
        self.assertEqual(svg, svg_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
