#!/usr/bin/env python3
import argparse
import json
import sys
from cerebra.orchestrator.super_agent import SuperAgent

def parse_args():
    parser = argparse.ArgumentParser(
        description="Orchestrate multi-modal patient-level disease prediction."
    )
    parser.add_argument(
        "--patient_id",
        type=str,
        required=True,
        help="Unique identifier for the patient.",
    )

    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="If set, serialize the full pipeline output to JSON file.",
    )

    parser.add_argument(
        "--llm_engine",
        type=str,
        default="gpt-4o",
        help="LLM engine to use for orchestration.",
    )

    parser.add_argument(
        "--year",
        type=int,
        default=1,
        help="Year of the data to use.",
    )

    parser.add_argument(
        "--institution",
        type=str,
        default="NYU",
        help="Institution to use.",
    )
    parser.add_argument(
        "--diagnosis",
        type=bool,
        default=False,
        help="Diagnosis to use.",
    )
    parser.add_argument(
        "--time_to_event",
        type=bool,
        default=False,
        help="Time to event to use.",
    )
    parser.add_argument(
        "--volume",
        type=bool,
        default=True,
        help="Volume to use.",
    )

    return parser.parse_args()
    
def main():
    args = parse_args()

    try:
        # Initialize SuperAgent
        orchestrator = SuperAgent(llm_engine_name=args.llm_engine)

        # Let SuperAgent handle everything including data loading
        result_metadata = orchestrator.run(
            task=f"Predict dementia risk after {args.year} years for patient id {args.patient_id}",
            patient_id=args.patient_id,
            year=args.year,
            institution=args.institution,
            diagnosis=args.diagnosis,
            time_to_event=args.time_to_event,
            volume=args.volume
        )

        # Extract and display results
        result_info = result_metadata.get_metadata_info()
        print("result_info: ", result_info)
        orchestration_result = result_info["dataset"].get("final_orchestration", {})

        print("\n=== Final Prediction Results ===\n")
        if isinstance(orchestration_result, dict):
            if "summary" in orchestration_result:
                print(orchestration_result["summary"])
            elif "response" in orchestration_result:
                print(orchestration_result["response"])
            else:
                print(json.dumps(orchestration_result, indent=2))
        else:
            print(str(orchestration_result))
        print("\n===============================\n")

        # Save output if requested
        if args.output_json:
            full_output = {
                "patient_id": args.patient_id,
                "prediction_result": orchestration_result,
                "result_metadata_info": result_info,
                "agents_used": result_info["dataset"].get("agents_used", [])
            }

            with open(args.output_json, "w") as f:
                json.dump(full_output, f, indent=2, default=str)
            print(f"Wrote full output to {args.output_json}")

    except Exception as e:
        print(f"Error during execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
