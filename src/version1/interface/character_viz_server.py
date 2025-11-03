import os
import json
from flask import Flask, render_template_string, request
from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import base64
from werkzeug.utils import secure_filename

app = Flask(__name__)

# --- Configuration ---
COCO_PREDICTIONS_JSON = "E:\\CalibreComics\\test_dections\\predictions.json"
CANONICAL_CSV = "C:\\Users\\Richard\\OneDrive\\GIT\\CoMix\\data_analysis\\key_mapping_report_claude.csv"
IMAGE_ROOT = "E:\\CalibreComics_extracted"
UPLOAD_FOLDER = "C:\\Users\\Richard\\OneDrive\\GIT\\CoMix\\temp_uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# --- Global Data Storage ---
coco_data = {}
coco_categories = {}
canonical_df = None

CATEGORY_COLORS = {
    'panel': 'yellow',
    'character': 'lime',
    'face': 'cyan',
    'balloon': 'orange',
    'text': 'red',
    'onomatopoeia': 'magenta',
    'link_sbsc': 'purple',
    'other': 'gray',
    'unknown': 'gray'
}

def load_data():
    global coco_data, coco_categories, canonical_df
    print("Loading canonical CSV...")
    try:
        canonical_df = pd.read_csv(CANONICAL_CSV)
        print(f"Loaded {len(canonical_df)} entries from canonical CSV.")
    except Exception as e:
        print(f"Error loading canonical CSV {CANONICAL_CSV}: {e}")
        canonical_df = pd.DataFrame()

    print("Loading COCO predictions JSON...")
    try:
        with open(COCO_PREDICTIONS_JSON, 'r', encoding='utf-8') as f:
            coco_raw_data = json.load(f)
        coco_categories = {cat['id']: cat['name'] for cat in coco_raw_data['categories']}
        for ann in coco_raw_data['annotations']:
            image_id = ann['image_id']
            if image_id not in coco_data:
                coco_data[image_id] = []
            coco_data[image_id].append(ann)
        print(f"Loaded {len(coco_data)} images with detections from COCO JSON.")
    except Exception as e:
        print(f"Error loading COCO predictions JSON {COCO_PREDICTIONS_JSON}: {e}")

with app.app_context():
    load_data()

# --- Visualization Function ---
def create_all_detections_visualization(image_path, detections):
    """Create visualization with all detection bounding boxes."""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    
    try:
        # Load image
        img = Image.open(image_path).convert('RGB')
        img = ImageOps.exif_transpose(img)
        W_orig, H_orig = img.size
        
        print(f"Original image size: {W_orig}x{H_orig}")
        print(f"Number of detections: {len(detections)}")
        
        # Resize image if too large to prevent memory issues
        max_dim = 800  # Reasonable size for visualization
        resize_scale = 1.0
        if max(W_orig, H_orig) > max_dim:
            resize_scale = max_dim / max(W_orig, H_orig)
            new_w = int(W_orig * resize_scale)
            new_h = int(H_orig * resize_scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            W, H = new_w, new_h
            print(f"Resized image to {W}x{H} (scale={resize_scale:.3f})")
        else:
            W, H = W_orig, H_orig
            print(f"Image size {W}x{H} is within limits")
        
        # Clean up existing figures
        plt.close('all')
        
        # Create figure with appropriate aspect ratio
        fig_width = 12
        fig_height = 12 * (H / W) if H > W else 12
        fig, ax = plt.subplots(1, figsize=(fig_width, fig_height), dpi=100)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title('Character & Text Bounding Boxes', fontsize=14, fontweight='bold', pad=10)
        
        # Draw bounding boxes
        for det in detections:
            bbox = det['bbox']  # COCO format: [x, y, w, h] in original image pixel space
            category_id = det['category_id']
            category_name = coco_categories.get(category_id, 'unknown')
            score = det.get('score', 0.0)
            color = CATEGORY_COLORS.get(category_name, 'gray')
            
            # Bbox is in original image pixel space - scale to display size
            x_orig, y_orig, w_orig, h_orig = bbox
            
            # Scale coordinates to display size
            x = x_orig * resize_scale
            y = y_orig * resize_scale
            w = w_orig * resize_scale
            h = h_orig * resize_scale
            
            # Draw rectangle
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                    edgecolor=color, facecolor='none', alpha=0.7)
            ax.add_patch(rect)
            
            # Add label
            label_text = f'{category_name} ({score:.2f})'
            ax.text(x, y - 5, label_text, color='white', fontsize=8, 
                    bbox=dict(facecolor=color, alpha=0.6, pad=1))
        
        # Convert plot to base64 image
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg', bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    except Exception as e:
        print(f"Error creating visualization for {image_path}: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')
        return None

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- HTML Template ---
HTML_TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Character Viz Server</title>
    <style>
        body { font-family: sans-serif; margin: 20px; background-color: #f4f4f4; }
        h1 { color: #333; }
        .controls { 
            margin-bottom: 20px; 
            padding: 15px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .controls input, .controls select { padding: 8px; margin-right: 10px; }
        .controls button { padding: 8px 15px; background-color: #007bff; color: white; border: none; border-radius: 3px; cursor: pointer; }
        .controls button:hover { background-color: #0056b3; }
        .section { margin-bottom: 20px; padding-bottom: 15px; border-bottom: 1px solid #ddd; }
        .section:last-child { border-bottom: none; }
        img { max-width: 100%; height: auto; border: 1px solid #ccc; margin-top: 10px; }
        .legend {
            margin-top: 10px;
            padding: 10px;
            background-color: white;
            border-radius: 5px;
        }
        .legend-item {
            display: inline-block;
            margin-right: 15px;
            padding: 3px 8px;
            border: 2px solid;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <h1>🎭 Character & Detection Visualizer</h1>
    
    <div class="controls">
        <!-- Index-based navigation -->
        <div class="section">
            <h3>📊 Browse by Canonical CSV Index</h3>
            <form action="/" method="get">
                <label for="index_input">Enter Index (0 to {{ total_images - 1 }}):</label><br>
                <input type="number" id="index_input" name="index" min="0" max="{{ total_images - 1 }}" 
                       value="{{ current_index }}" style="width: 150px;">
                <button type="submit">View by Index</button>
            </form>
        </div>
        
        <!-- File upload -->
        <div class="section">
            <h3>📁 Upload Custom Image</h3>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".jpg,.jpeg,.png,.gif,.bmp" required>
                <button type="submit">Upload & Visualize</button>
            </form>
            <p style="font-size: 0.9em; color: #666;">Supported formats: JPG, PNG, GIF, BMP</p>
        </div>
        
        {% if image_path %}
        <div class="section">
            <p><strong>Current Image:</strong> <code>{{ image_path }}</code></p>
            <p><strong>Detections Found:</strong> {{ num_detections }}</p>
            {% if detection_info %}
            <details>
                <summary style="cursor: pointer; font-weight: bold;">📋 Detection Details (click to expand)</summary>
                <pre style="background-color: #f8f8f8; padding: 10px; border-radius: 3px; overflow-x: auto; font-size: 0.85em;">{{ detection_info }}</pre>
            </details>
            {% endif %}
        </div>
        {% endif %}
    </div>
    
    <!-- Legend -->
    <div class="legend">
        <strong>Detection Categories:</strong><br>
        <span class="legend-item" style="border-color: yellow;">Panel</span>
        <span class="legend-item" style="border-color: lime;">Character</span>
        <span class="legend-item" style="border-color: cyan;">Face</span>
        <span class="legend-item" style="border-color: orange;">Balloon</span>
        <span class="legend-item" style="border-color: red;">Text</span>
        <span class="legend-item" style="border-color: magenta;">Onomatopoeia</span>
    </div>

    {% if viz_image %}
    <img src="data:image/jpeg;base64,{{ viz_image }}" alt="Visualization of {{ image_path }}">
    {% elif error %}
    <p style="color: red; font-weight: bold;">❌ {{ error }}</p>
    {% endif %}
</body>
</html>
'''

# --- Flask Routes ---
@app.route('/')
def index():
    """Main route for index-based browsing."""
    index_str = request.args.get('index', '0')
    total_images = len(canonical_df) if canonical_df is not None else 0
    
    try:
        index = int(index_str)
    except ValueError:
        return render_template_string(HTML_TEMPLATE, 
                                     error="Invalid index. Please enter a number.", 
                                     total_images=total_images,
                                     current_index=0)

    if canonical_df is None or index < 0 or index >= len(canonical_df):
        return render_template_string(HTML_TEMPLATE, 
                                     error=f"Index out of bounds. Max index is {len(canonical_df) - 1 if canonical_df is not None else 0}.", 
                                     total_images=total_images,
                                     current_index=0)

    image_path = canonical_df.iloc[index]['image_path']
    
    # Get the predictions_json_id from the CSV - this is the exact key used in predictions.json
    predictions_json_id = canonical_df.iloc[index]['predictions_json_id']
    
    # Look up detections using the predictions_json_id as key
    detections = coco_data.get(predictions_json_id, [])
    
    print(f"======================================================================")
    print(f"DEBUG: Looking up detections")
    print(f"Image path from CSV: {repr(image_path)}")
    print(f"Predictions JSON ID from CSV: {repr(predictions_json_id)}")
    print(f"Found {len(detections)} detections using predictions_json_id")
    
    # If still no detections found, report the mismatch
    if len(detections) == 0:
        print(f"WARNING: No detections found for predictions_json_id: {predictions_json_id}")
        # Check what keys are available in coco_data
        sample_keys = list(coco_data.keys())[:5]
        print(f"Sample COCO data keys (showing repr for comparison):")
        for k in sample_keys:
            print(f"  {repr(k)}")
    
    print(f"Final detection count: {len(detections)}")
    
    # Print detection details for debugging
    detection_info = ""
    if len(detections) > 0:
        print(f"Sample detection (first one):")
        print(f"  Category ID: {detections[0]['category_id']}")
        print(f"  Category Name: {coco_categories.get(detections[0]['category_id'], 'unknown')}")
        print(f"  Bbox: {detections[0]['bbox']}")
        print(f"  Score: {detections[0].get('score', 'N/A')}")
        
        # Create detection info string for display
        detection_info = "COCO Predictions Info:\n\n"
        detection_info += f"Total Detections: {len(detections)}\n\n"
        
        # Group by category
        category_counts = {}
        for det in detections:
            cat_name = coco_categories.get(det['category_id'], 'unknown')
            category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
        
        detection_info += "Breakdown by Category:\n"
        for cat, count in sorted(category_counts.items()):
            detection_info += f"  - {cat}: {count}\n"
        
        detection_info += f"\nFirst 5 Detections (showing raw data from predictions.json):\n"
        for i, det in enumerate(detections[:5]):
            cat_name = coco_categories.get(det['category_id'], 'unknown')
            bbox = det['bbox']
            score = det.get('score', 0.0)
            detection_info += f"\n  Detection {i+1}:\n"
            detection_info += f"    Category: {cat_name}\n"
            detection_info += f"    Bbox (COCO x,y,w,h): {bbox}\n"
            detection_info += f"    Score: {score:.3f}\n"
            detection_info += f"    Raw JSON: {json.dumps(det, indent=6)}\n"
    else:
        detection_info = "No detections found for this image in predictions.json\n\n"
        detection_info += "This could mean:\n"
        detection_info += "  1. The image_id in predictions.json doesn't match the path format\n"
        detection_info += "  2. No detections were found for this image during prediction\n"
        detection_info += f"\nSample keys from predictions.json:\n"
        sample_keys = list(coco_data.keys())[:5]
        for key in sample_keys:
            detection_info += f"  - {key}\n"
    print(f"======================================================================")
    
    viz_image_base64 = create_all_detections_visualization(image_path, detections)

    if viz_image_base64 is None:
        return render_template_string(HTML_TEMPLATE, 
                                     error=f"Could not generate visualization for image: {image_path}", 
                                     total_images=total_images,
                                     current_index=index,
                                     image_path=image_path,
                                     num_detections=len(detections),
                                     detection_info=detection_info)

    return render_template_string(HTML_TEMPLATE, 
                                 image_path=image_path, 
                                 viz_image=viz_image_base64, 
                                 total_images=total_images,
                                 current_index=index,
                                 num_detections=len(detections),
                                 detection_info=detection_info)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and visualization."""
    total_images = len(canonical_df) if canonical_df is not None else 0
    
    if 'file' not in request.files:
        return render_template_string(HTML_TEMPLATE, 
                                     error="No file uploaded.", 
                                     total_images=total_images,
                                     current_index=0)
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template_string(HTML_TEMPLATE, 
                                     error="No file selected.", 
                                     total_images=total_images,
                                     current_index=0)
    
    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"Uploaded file saved to: {filepath}")
        
        # For uploaded files, we won't have detections in COCO data
        # So we'll just visualize the image with empty detections
        detections = []
        detection_info = "No detections available for uploaded images.\n"
        detection_info += "Detections are only available for images in the canonical dataset."
        
        viz_image_base64 = create_all_detections_visualization(filepath, detections)
        
        if viz_image_base64 is None:
            return render_template_string(HTML_TEMPLATE, 
                                         error=f"Could not generate visualization for uploaded image.", 
                                         total_images=total_images,
                                         current_index=0)
        
        return render_template_string(HTML_TEMPLATE, 
                                     image_path=filepath, 
                                     viz_image=viz_image_base64, 
                                     total_images=total_images,
                                     current_index=0,
                                     num_detections=len(detections),
                                     detection_info=detection_info)
    else:
        return render_template_string(HTML_TEMPLATE, 
                                     error="Invalid file type. Please upload JPG, PNG, GIF, or BMP.", 
                                     total_images=total_images,
                                     current_index=0)

if __name__ == '__main__':
    app.run(debug=True, port=5003)