import os
import sys
import importlib
import inspect
import traceback
from typing import Dict, Any, List, Tuple
import time
class Initializer:
    def __init__(self, agent_name: str, enabled_tools: List[str] = [], model_string: str = None, verbose: bool = False, vllm_config_path: str = None):
        self.toolbox_metadata = {}
        self.available_tools = []
        self.enabled_tools = enabled_tools
        self.load_all = self.enabled_tools == ["all"]
        self.model_string = model_string # llm model string
        self.verbose = verbose
        self.vllm_server_process = None
        self.vllm_config_path = vllm_config_path
        self.agent_name = agent_name
        print(f"\n==> Initializing {self.agent_name}...")
        print(f"Enabled tools: {self.enabled_tools}")
        print(f"LLM engine name: {self.model_string}")
        self._set_up_tools()
        
        # if vllm, set up the vllm server
        if model_string.startswith("vllm-"):
            self.setup_vllm_server()

    def get_project_root(self):
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        while current_dir != '/':
            if os.path.exists(os.path.join(current_dir)):
                return os.path.join(current_dir)
            current_dir = os.path.dirname(current_dir)
        raise Exception("Could not find project root")
        
    def load_tools_and_get_metadata(self) -> Dict[str, Any]:
        # Implementation of load_tools_and_get_metadata function
        self.toolbox_metadata = {}
        project_dir = self.get_project_root()
        tools_dir = os.path.join(project_dir, 'tools', self.agent_name)  

        sys.path.insert(0, project_dir)
        sys.path.insert(0, os.path.dirname(project_dir))
        # print(f"Updated Python path: {sys.path}")
        
        if not os.path.exists(tools_dir):
            print(f"Error: Tools directory does not exist: {tools_dir}")
            return self.toolbox_metadata   

        for root, dirs, files in os.walk(tools_dir):
            if 'tool.py' in files and (self.load_all or os.path.basename(root) in self.available_tools):
                file = 'tool.py'
                module_path = os.path.join(root, file)
                module_name = os.path.splitext(file)[0]
                relative_path = os.path.relpath(module_path, project_dir)
                import_path = '.'.join(os.path.split(relative_path)).replace(os.sep, '.')[:-3]
               
                print(f"\n==> Attempting to import: {import_path}")
                try:
                    module = importlib.import_module(import_path)
                    for name, obj in inspect.getmembers(module):
                        if inspect.isclass(obj) and name.endswith('Tool') and name != 'BaseTool':
                            print(f"Found tool class: {name}")
                            
                            try:
                                # Check if the tool requires an LLM engine
                                if hasattr(obj, 'require_llm_engine') and obj.require_llm_engine:
                                    tool_instance = obj(model_string=self.model_string)
                                else:
                                    tool_instance = obj()
                                
                                self.toolbox_metadata[name] = {
                                    'tool_name': getattr(tool_instance, 'tool_name', 'Unknown'),
                                    'tool_description': getattr(tool_instance, 'tool_description', 'No description'),
                                    'tool_version': getattr(tool_instance, 'tool_version', 'Unknown'),
                                    'input_types': getattr(tool_instance, 'input_types', {}),
                                    'output_type': getattr(tool_instance, 'output_type', 'Unknown'),
                                    'demo_commands': getattr(tool_instance, 'demo_commands', []),
                                    'user_metadata': getattr(tool_instance, 'user_metadata', {}), # This is a placeholder for user-defined metadata
                                    'evaluation_criteria': getattr(tool_instance, 'evaluation_criteria', {}),
                                    'require_llm_engine': getattr(obj, 'require_llm_engine', False),
                                    'import_path': import_path,
                                }
                                # print(f"Metadata for {name}: {self.toolbox_metadata[name]}")
                            except Exception as e:
                                print(f"Error instantiating {name}: {str(e)}")
                except Exception as e:
                    print(f"Error loading module {module_name}: {str(e)}")
                    
        print(f"\n==> Total number of tools imported: {len(self.toolbox_metadata)}")

        return self.toolbox_metadata

    def run_demo_commands(self) -> List[str]:
        self.available_tools = []

        for tool_name, tool_data in self.toolbox_metadata.items():
            print(f"Checking availability of {tool_name}...")

            try:
                # Use the import path recorded during loading (avoids CamelCase → path bugs)
                module_name = tool_data.get('import_path', f"tools.{self.agent_name}.{tool_name.lower()}.tool")
                module = importlib.import_module(module_name)

                # Get the tool class
                tool_class = getattr(module, tool_name)

                # Instantiate the tool
                tool_instance = tool_class()

                # FIXME This is a temporary workaround to avoid running demo commands
                self.available_tools.append(tool_name)

            except Exception as e:
                print(f"Error checking availability of {tool_name}: {str(e)}")
                print(traceback.format_exc())

        # update the toolmetadata with the available tools
        self.toolbox_metadata = {tool: self.toolbox_metadata[tool] for tool in self.available_tools}
        # print(f"Updated total number of available tools: {len(self.toolbox_metadata)}")
        # print(f"Available tools: {self.available_tools}")
        return self.available_tools
    
    def _set_up_tools(self) -> None:
        print(f"\n==> Setting up tools for {self.agent_name}...")

        # Keep enabled tools
        self.available_tools = [tool.lower().replace('_tool', '') for tool in self.enabled_tools]
        # Load tools and get metadata
        self.load_tools_and_get_metadata()
        
        # Run demo commands to determine available tools
        self.run_demo_commands()
        
        # Filter toolbox_metadata to include only available tools
        self.toolbox_metadata = {tool: self.toolbox_metadata[tool] for tool in self.available_tools}
        print(f"✅ Total number of final available tools: {len(self.available_tools)}")
        print(f"✅ Final available tools: {self.available_tools}")

    def setup_vllm_server(self) -> None:
        # Check if vllm is installed
        try:
            import vllm
        except ImportError:
            raise ImportError("If you'd like to use VLLM models, please install the vllm package by running `pip install vllm`.")
        
        # Start the VLLM server
        command = ["vllm", "serve", self.model_string.replace("vllm-", ""), "--port", "8888"]
        if self.vllm_config_path is not None:
            command = ["vllm", "serve", "--config", self.vllm_config_path, "--port", "8888"]

        import subprocess
        vllm_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        print("Starting VLLM server...")
        while True:
            output = vllm_process.stdout.readline()
            error = vllm_process.stderr.readline()
            time.sleep(5)
            if output.strip() != "":
                print("VLLM server standard output:", output.strip())
            if error.strip() != "":
                print("VLLM server standard error:", error.strip())

            if "Application startup complete." in output or "Application startup complete." in error:
                print("VLLM server started successfully.")
                break

            if vllm_process.poll() is not None:
                print("VLLM server process terminated unexpectedly. Please check the output above for more information.")
                break

        self.vllm_server_process = vllm_process

if __name__ == "__main__":
    enabled_tools = ["ehr_feature_extractor"]
    initializer = Initializer(agent_name="ehr_agent", enabled_tools=enabled_tools, model_string="gpt-4o")

    print("\nAvailable tools:")
    print(initializer.available_tools)

    print("\nToolbox metadata for available tools:")
    print(initializer.toolbox_metadata)
    