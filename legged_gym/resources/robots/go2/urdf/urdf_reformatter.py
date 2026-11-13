import xml.etree.ElementTree as ET

# Function to reformat the URDF file
def reformat_urdf(input_file, output_file):
    # Parse the URDF file
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Helper function to format elements
    def format_element(elem, level=0):
        indent = "    " * level
        text = f"<{elem.tag}"
        
        # Add attributes inline
        if elem.attrib:
            for key, value in elem.attrib.items():
                text += f' {key}="{value}"'
        
        # Check for children
        if list(elem):
            text += ">\n"
            for child in elem:
                text += format_element(child, level + 1)
            text += f"{indent}</{elem.tag}>\n"
        else:
            text += " />\n"

        return indent + text

    # Build the formatted content
    formatted_content = "<?xml version=\"1.0\"?>\n" + format_element(root)

    # Write to the output file
    with open(output_file, "w") as f:
        f.write(formatted_content)

# Input and output file paths
input_file = "/home/yulong/Github/legged_gym_go2/legged_gym/resources/robots/go2/urdf/go2_no_mesh.urdf"
output_file = "/home/yulong/Github/legged_gym_go2/legged_gym/resources/robots/go2/urdf/go2_no_mesh.urdf"

# Reformat the URDF
reformat_urdf(input_file, output_file)

print(f"Reformatted URDF has been written to {output_file}")
