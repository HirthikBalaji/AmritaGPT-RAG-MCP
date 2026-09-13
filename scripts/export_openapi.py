"""
Export OpenAPI 3.1 Specification for AmritaGPT.
Generates openapi.json and openapi.yaml in the project root.
"""
import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amritagpt.api.app import app


def main():
    openapi_schema = app.openapi()
    
    root_dir = Path(__file__).resolve().parent.parent
    json_path = root_dir / "openapi.json"
    yaml_path = root_dir / "openapi.yaml"

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(openapi_schema, f, indent=2)
    print(f"Exported OpenAPI JSON to: {json_path}")

    # Save YAML if pyyaml is installed
    try:
        import yaml
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(openapi_schema, f, sort_keys=False)
        print(f"Exported OpenAPI YAML to: {yaml_path}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
