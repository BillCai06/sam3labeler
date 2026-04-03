#!/usr/bin/env python3
"""
CLI entry point for batch processing.

Usage:
    python run_batch.py --input /path/to/images --classes car person truck
    python run_batch.py --input ./frames/ --classes car person --config config.yaml
    python run_batch.py --input ./frames/ --classes car person --no-viz
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 batch detection + segmentation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", default=None,
        help="Path to image folder or single image file (required for CLI batch mode)",
    )
    parser.add_argument(
        "--classes", "-c", nargs="+", default=None,
        help="Class names to detect (required for CLI batch mode)",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output directory (default: outputs/run_TIMESTAMP)",
    )
    parser.add_argument(
        "--confidence", type=float, default=None,
        help="Override confidence threshold from config",
    )
    parser.add_argument(
        "--sam-score", type=float, default=None,
        help="Override SAM score threshold from config",
    )
    parser.add_argument(
        "--no-viz", action="store_true",
        help="Skip saving visualization images",
    )
    parser.add_argument(
        "--no-coco", action="store_true",
        help="Skip saving COCO JSON",
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Launch Gradio GUI instead of CLI batch mode",
    )
    parser.add_argument(
        "--api", action="store_true",
        help="Launch FastAPI backend instead of CLI batch mode",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Port for GUI or API server",
    )

    args = parser.parse_args()

    # GUI mode
    if args.gui:
        from gui.app import build_app
        from src.utils import load_config
        config = load_config(args.config)
        port = args.port or config.get("gui", {}).get("port", 7860)
        host = config.get("gui", {}).get("host", "0.0.0.0")
        demo = build_app(config_path=args.config)
        demo.launch(server_name=host, server_port=port, allowed_paths=["/mnt/usbssd", "/tmp"])
        return

    # API mode
    if args.api:
        import uvicorn
        from src.utils import load_config
        config = load_config(args.config)
        port = args.port or config.get("api", {}).get("port", 8000)
        host = config.get("api", {}).get("host", "0.0.0.0")
        uvicorn.run("api.main:app", host=host, port=port, reload=False)
        return

    # CLI batch mode — require input and classes
    if not args.input:
        parser.error("--input/-i is required for CLI batch mode")
    if not args.classes:
        parser.error("--classes/-c is required for CLI batch mode")

    from src.utils import load_config
    from src.pipeline import Pipeline
    from src.batch_processor import BatchProcessor

    config = load_config(args.config)

    # Override thresholds if provided
    if args.confidence is not None:
        config.setdefault("pipeline", {})["confidence_threshold"] = args.confidence
    if args.sam_score is not None:
        config.setdefault("pipeline", {})["sam_score_threshold"] = args.sam_score

    print(f"\n{'='*60}")
    print(f"SAM3 Batch Processor")
    print(f"Input:   {args.input}")
    print(f"Classes: {args.classes}")
    print(f"Config:  {args.config}")
    print(f"{'='*60}\n")

    pipeline = Pipeline(config)
    batch = BatchProcessor(pipeline, config)

    summary = batch.run(
        input_path=args.input,
        active_classes=args.classes,
        output_dir=args.output,
        save_viz=not args.no_viz,
        save_coco=not args.no_coco,
    )

    if summary.get("coco_path"):
        print(f"\nCOCO JSON: {summary['coco_path']}")

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
