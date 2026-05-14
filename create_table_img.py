import pandas as pd
import matplotlib.pyplot as plt
import os

def render_table_to_image(csv_path: str, output_path: str):
    """
    Reads a CSV file containing descriptive statistics and renders it 
    as a nicely formatted table image using matplotlib.
    """
    # 1. Read data
    df = pd.read_csv(csv_path, header=None)
    df = df.fillna('')  # Replace NaN with empty strings

    # 2. Setup figure
    fig, ax = plt.subplots(figsize=(24, 10))
    ax.axis('tight')
    ax.axis('off')

    # 3. Create table
    table_data = df.values
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    
    # Configure basic table properties
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    # 4. Style table cells
    for (row_idx, col_idx), cell in table.get_celld().items():
        # First column (Variables): Left align text for data rows
        if col_idx == 0 and row_idx > 1:
            cell.set_text_props(ha='left')
            
        # Headers (Rows 0 and 1)
        if row_idx in [0, 1]:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2C3E50')
            
        # First column styling (Variable names)
        elif col_idx == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#ECF0F1')
            
        # Data cells styling
        else:
            # Format numbers to 3 decimal places
            try:
                val = float(cell.get_text().get_text())
                cell.get_text().set_text(f"{val:.3f}")
            except ValueError:
                pass
                
            # Alternate row colors for better readability
            if row_idx % 2 == 0:
                cell.set_facecolor('#F9F9F9')
            else:
                cell.set_facecolor('#FFFFFF')

    # 5. Save image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Table image successfully saved to {output_path}")

if __name__ == '__main__':
    csv_file = 'outputs/descriptive_stats.csv'
    out_file = 'outputs/descriptive_stats.png'
    render_table_to_image(csv_file, out_file)
