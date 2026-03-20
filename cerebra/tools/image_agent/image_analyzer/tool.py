import os
import pickle
import numpy as np
import torch
from PIL import Image
from typing import List, Optional, Dict, Union, Any
from transformers import AutoProcessor, AutoModelForImageTextToText
from cerebra.tools.base import BaseTool
from cerebra.utils.dataset import Dataset
from cerebra.utils.log_utils import setup_logger

# Configure logging to save to the tool's cache directory
log_dir = os.path.join("cerebra_cache", "image_agent", "logs")
os.makedirs(log_dir, exist_ok=True)
# Setup logger
logger = setup_logger(log_dir)

class MRI_Analyzer_Tool(BaseTool):
    """
    A tool that analyzes MRI scans (2D slices) using the MedGemma multimodal language model.
    Generates comprehensive analysis queries for brain structure assessment and disease likelihood prediction.
    """
    
    def __init__(self):
        super().__init__()
        
        # Output directories
        self.analysis_output_dir = os.path.join("cerebra_cache", "image_agent", "analyses")
        self.models_cache_dir = os.path.join("cerebra_cache", "image_agent", "models")
        os.makedirs(self.analysis_output_dir, exist_ok=True)
        os.makedirs(self.models_cache_dir, exist_ok=True)

        self.set_metadata(
            tool_name="MRI_Analyzer_Tool",
            tool_description="Analyzes MRI scans using MedGemma multimodal model to assess brain structure and disease likelihood",
            tool_version="1.0.0",
            input_types={
                "image_paths(required)": "Union[str, List[str]] - Path(s) to MRI scan images (PNG format)",
                "patient_ids": "Union[str, List[str]] - Patient identifier(s) corresponding to the images",
                "custom_queries": "List[str] - Additional custom analysis queries to run",
                "include_disease_assessment": "bool - Whether to include disease likelihood assessment (default: True)",
                "include_structure_analysis": "bool - Whether to include brain structure analysis (default: True)",
                "save_name": "str - Name for saving the analysis results (default: 'mri_analysis')"
            },
            output_type="Dataset - Dataset object containing comprehensive MRI analysis results",
            demo_commands=[
                {
                    "command": "result = tool.execute(image_paths=['scan1.png', 'scan2.png'], patient_ids=['P001', 'P002'], save_name='brain_analysis')",
                    "description": "Analyze multiple MRI scans with comprehensive brain structure and disease assessment"
                },
                {
                    "command": "result = tool.execute(image_paths='single_scan.png', custom_queries=['Assess for signs of stroke', 'Evaluate white matter lesions'])",
                    "description": "Analyze single MRI scan with custom queries"
                }
            ],
            user_metadata={
                "limitations": [
                    "Requires GPU for optimal performance",
                    "Currently supports PNG format images",
                    "Model responses are for research purposes, not clinical diagnosis",
                    "Processing time scales with number of images and queries"
                ],
                "best_practices": [
                    "Ensure MRI images are properly preprocessed and oriented",
                    "Use consistent naming conventions for patient IDs",
                    "Include relevant clinical context in custom queries",
                    "Validate results with medical professionals",
                    "Monitor GPU memory usage for large batch processing"
                ]
            }
        )

        # Initialize model components
        self.model = None
        self.processor = None
        self.device = None

    def _initialize_model(self):
        """Initialize the MedGemma model and processor."""
        if self.model is None:
            logger.info("Initializing MedGemma model...")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            model_id = "google/medgemma-4b-it"
            
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir=self.models_cache_dir
            )
            self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=self.models_cache_dir)
            
            logger.info(f"MedGemma model loaded successfully on device: {self.device}")

    def _load_image(self, image_path: str) -> Image.Image:
        """Load and validate MRI image."""
        try:
            image = Image.open(image_path)
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            raise ValueError(f"Could not load image from {image_path}: {str(e)}")

    def _generate_standard_queries(self, include_disease_assessment: bool = True, 
                                 include_structure_analysis: bool = True) -> List[str]:
        """Generate standard analysis queries for MRI scans."""
        queries = []
        
        if include_structure_analysis:
            structure_queries = [
                "Describe the overall brain anatomy visible in this MRI scan",
                "Assess the gray matter and white matter distribution",
                "Evaluate the ventricular system and CSF spaces",
                "Examine the cortical thickness and any atrophy patterns",
                "Describe the hippocampal structures and their appearance",
                "Assess the brainstem and cerebellar structures",
                "Evaluate any visible vascular structures or abnormalities"
            ]
            queries.extend(structure_queries)

        if include_disease_assessment:
            disease_queries = [
                "Assess this MRI scan for signs of Alzheimer's disease or dementia",
                "Evaluate for any signs of stroke or cerebrovascular disease",
                "Look for signs of multiple sclerosis or white matter lesions",
                "Assess for any tumors or space-occupying lesions",
                "Evaluate for signs of traumatic brain injury",
                "Look for signs of movement disorders",
                "Assess for any developmental or congenital abnormalities",
                "Evaluate the overall likelihood of neurodegenerative disease"
            ]
            queries.extend(disease_queries)

        return queries

    def _analyze_image_with_query(self, image: Image.Image, query: str, patient_id: str = None) -> Dict:
        """Analyze a single image with a specific query."""
        try:
            # Prepare system message
            system_message = "You are an expert neuroradiologist analyzing MRI brain scans. Provide detailed, accurate observations based on what you can see in the image. Include confidence levels in your assessments."
            
            # Prepare messages for the model
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {"type": "image", "image": image}
                    ]
                }
            ]

            # Process inputs
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16)

            input_len = inputs["input_ids"].shape[-1]

            # Generate response
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs, 
                    max_new_tokens=300, 
                    do_sample=False,
                    # temperature=0.1
                )
                generation = generation[0][input_len:]

            # Decode response
            response = self.processor.decode(generation, skip_special_tokens=True)
            
            return response.split("Here's my assessment:")[1].strip()

        except Exception as e:
            logger.error(f"Error analyzing image with query '{query}': {str(e)}")
            return {
                "query": query,
                "response": f"Error during analysis: {str(e)}",
                "patient_id": patient_id,
                "status": "error"
            }

    def _process_single_image(self, image_path: str, patient_id: str, queries: List[str]) -> Dict:
        """Process a single MRI image with all queries."""
        logger.info(f"Processing image: {image_path} for patient: {patient_id}")
        
        # Load image
        image = self._load_image(image_path)
        
        # Run analysis for each query
        analyses = []
        for i, query in enumerate(queries):
            logger.info(f"Running query {i+1}/{len(queries)}: {query[:50]}...")
            analysis = self._analyze_image_with_query(image, query + ".\nYour response should start with \"Here's my assessment:\"", patient_id)
            analyses.append(analysis)

        return {
            "patient_id": patient_id,
            "image_path": image_path,
            "queries": queries,
            "analyses": analyses,
            "total_queries": len(queries)
        }

    def _save_results(self, results: Dict, save_name: str) -> str:
        """Save analysis results to file."""
        save_path = os.path.join(self.analysis_output_dir, f"{save_name}_results.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"Results saved to: {save_path}")
        return save_path

    def execute(self, image_paths: Union[str, List[str]], 
                patient_ids: Union[str, List[str]] = None,
                custom_queries: List[str] = None,
                include_disease_assessment: bool = True,
                include_structure_analysis: bool = True,
                save_name: str = "mri_analysis", **kwargs):
        """
        Execute MRI analysis using MedGemma multimodal model.

        Args:
            image_paths: Path(s) to MRI scan images (PNG format)
            patient_ids: Patient identifier(s) corresponding to the images
            custom_queries: Additional custom analysis queries to run
            include_disease_assessment: Whether to include disease likelihood assessment
            include_structure_analysis: Whether to include brain structure analysis
            save_name: Name for saving the analysis results

        Returns:
            Dataset: Dataset object containing comprehensive MRI analysis results
        """
        try:
            # Initialize model
            self._initialize_model()
            
            # Normalize inputs to lists
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            
            if patient_ids is None:
                patient_ids = [f"Patient_{i+1}" for i in range(len(image_paths))]
            elif isinstance(patient_ids, str):
                patient_ids = [patient_ids]
            
            if len(image_paths) != len(patient_ids):
                raise ValueError("Number of image paths must match number of patient IDs")

            # Generate queries
            standard_queries = self._generate_standard_queries(
                include_disease_assessment, include_structure_analysis
            )
            
            all_queries = standard_queries.copy()
            if custom_queries:
                all_queries.extend(custom_queries)

            logger.info(f"Starting analysis of {len(image_paths)} images with {len(all_queries)} queries each")

            # Process each image
            all_results = []
            for image_path, patient_id in zip(image_paths, patient_ids):
                result = self._process_single_image(image_path, patient_id, all_queries)
                all_results.append(result)

            # Compile final results
            final_results = {
                "analysis_summary": {
                    "total_images": len(image_paths),
                    "total_queries_per_image": len(all_queries),
                    "standard_queries_count": len(standard_queries),
                    "custom_queries_count": len(custom_queries) if custom_queries else 0,
                    "include_disease_assessment": include_disease_assessment,
                    "include_structure_analysis": include_structure_analysis
                },
                "patient_analyses": all_results,
                "query_list": all_queries,
                "model_info": {
                    "model_name": "google/medgemma-4b-it",
                    "model_type": "multimodal_language_model",
                    "device": str(self.device)
                },
                "status": "success"
            }
           

            # Save results
            results_path = self._save_results(final_results, save_name)

            # Create feature descriptions
            feature_descriptions = {
                "analysis_summary": "Overview of the analysis including image count and query statistics",
                "patient_analyses": """Detailed analysis results for each patient. 
                Usage: Access individual patient results with: results['patient_analyses'][patient_index]['analyses']
                Each analysis contains: query, response, patient_id, and status""",
                "query_list": "Complete list of queries used in the analysis",
                "model_info": "Information about the MedGemma model used for analysis",
                "results_file_path": f"""Path to saved detailed results file.
                Usage: Load full results with: pickle.load(open('{results_path}', 'rb'))""",
                "status": "Overall execution status"
            }

            # Add results file path to output data
            output_data = final_results.copy()
            output_data["results_file_path"] = results_path

            # Create Dataset object
            result_dataset = Dataset.create_agent_output(
                processed_data=output_data,
                description=f"MRI analysis using MedGemma model for {len(image_paths)} scans with {len(all_queries)} queries each. Includes brain structure analysis and disease likelihood assessment.",
                feature_descriptions=feature_descriptions,
                cache_directory=os.path.join("cerebra_cache", "image_agent")
            )

            logger.info(f"MRI analysis completed successfully!")
            logger.info(f"Analyzed {len(image_paths)} images with {len(all_queries)} queries each")
            logger.info(f"Results saved to: {results_path}")

            return result_dataset

        except Exception as e:
            logger.error(f"Error during MRI analysis: {str(e)}")
            
            # Return error as Dataset for consistency
            error_data = {
                "status": "error",
                "error_message": str(e),
                "parameters": {
                    "image_paths_provided": image_paths is not None,
                    "patient_ids_provided": patient_ids is not None,
                    "custom_queries_provided": custom_queries is not None,
                    "include_disease_assessment": include_disease_assessment,
                    "include_structure_analysis": include_structure_analysis,
                    "save_name": save_name
                }
            }
            
            error_descriptions = {
                "status": "Analysis execution status (error)",
                "error_message": "Description of the error that occurred",
                "parameters": "Parameters used when error occurred"
            }
            
            return Dataset.create_agent_output(
                processed_data=error_data,
                description=f"Error during MRI analysis with MedGemma model",
                feature_descriptions=error_descriptions,
                cache_directory=os.path.join("cerebra_cache", "image_agent")
            )


if __name__ == "__main__":
    # Test the MRI analyzer tool
    tool = MRI_Analyzer_Tool()
    
    # Note: For testing, you would need actual MRI image files
    # This is a demonstration of how to use the tool
    
    print("MRI Analyzer Tool Test")
    print("Note: This test requires actual MRI image files to run")
    
    # Example usage (commented out since we don't have actual image files):
    # """
    result = tool.execute(
        image_paths=["path/to/mri_slice.png"],
        patient_ids=["TEST_001"],
        custom_queries=["Describe this MRI slice"],
        save_name="test_mri_analysis",
        include_structure_analysis=False
    )
    
    # Extract results
    result_data = result.get_dataset()["dataset"]
    if result_data['status'] == 'success':
        print("✅ MRI analysis completed successfully!")
        print(f"Results saved to: {result_data['results_file_path']}")
        print(f"Analyzed {result_data['analysis_summary']['total_images']} images")
        print(f"Queries per image: {result_data['analysis_summary']['total_queries_per_image']}")
    else:
        print(f"❌ Analysis failed: {result_data.get('error_message', 'Unknown error')}")
    # """
