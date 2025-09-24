import tensorflow as tf
import numpy as np
import cv2, os
import io
import base64
from PIL import Image
import matplotlib.cm as cm
from tensorflow.keras.models import Model
from .model_utils import get_model
from .config import class_names



model = get_model()
base_model = model.get_layer("mobilenetv2_1.00_224")

def compute_gradcam(img_array, class_index, conv_layer_name):
    grad_model = Model(
        inputs = base_model.input,
        outputs = [
            base_model.get_layer(conv_layer_name).output, base_model.output
            ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)   # compute gradient
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))   # Global average pooling
    conv_outputs = conv_outputs.numpy()[0]
    pooled_grads = pooled_grads.numpy()


    # multiply feature maps by importance weights
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)   # compute heatmap
    heatmap = np.maximum(heatmap, 0)   # ReLU activation
    if (np.max(heatmap) != 0):
        heatmap /= np.max(heatmap)
    else: 
        heatmap = 1
    
    return heatmap



def overlay_heatmap(img_path, heatmap, alpha=0.6):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: Could not read image from {img_path}")
        return None
    
    img = cv2.cvtColor(img, cv2.IMREAD_COLOR)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = np.uint8(255 * heatmap)   # convert to 0-255 scale
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, alpha, heatmap, 1-alpha, 0)     # blend images
    return superimposed_img


def generate_gradcam(img_path, img_array):
    # get model predictions
    preds= model.predict(img_array)
    pred_index = np.argmax(preds[0])
    pred_class = class_names[pred_index]
    confidence = float(preds[0][pred_index])
    
    # confidence dict for all classes
    all_confidences = {
        class_names[i]: float(preds[0][i]) for i in range(len(class_names))
    }

    print(f"Prediction: {pred_class} (Index: {pred_index})")

    # compute gradcam heatmap
    heatmap = compute_gradcam(img_array, pred_index, conv_layer_name="Conv_1")
    print(np.max(heatmap), np.min(heatmap))

    # overlay heatmap on original image
    output_img = overlay_heatmap(img_path, heatmap)
    output_dir = "heatmap"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    if output_img is not None:
        base_filename = os.path.basename(img_path)    # extract filename from full path
        filename_without_ext = os.path.splitext(base_filename)[0]

        cv2.imwrite(os.path.join(output_dir, f"gradcam_overlay_{filename_without_ext}.jpg"), cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))

        
        just_heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, f"just_heatmap_color_{filename_without_ext}.jpg"), just_heatmap)

        print(f"Heatmaps saved to '{output_dir}/' directory")


        # convert img to BGR for cv2.imencode, if output_img is RGV
        output_img_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

        # encode the image to a byte buffer
        success, encoded_image = cv2.imencode('.png', output_img_bgr)
        if not success:
            # Handle the error if encoding fails
            print("Error: Could not encode output_img for base64.")
            return {
                "prediction": pred_class,
                "confidence": confidence,
                "all_confidences": all_confidences,
                "gradcam": None,
            }


        # conver to base64
        img_str = base64.b64encode(encoded_image.tobytes()).decode('utf-8')

        return {
            "prediction": pred_class,
            "confidence": confidence,
            "all_confidences": all_confidences, 
            "gradcam": img_str
        }

    else:
        print("Failed to generate overlay image")






