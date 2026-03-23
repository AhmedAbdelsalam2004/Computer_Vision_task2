import os
import uuid
import base64
import traceback
from io import BytesIO
import numpy as np
import cv2
from PIL import Image
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework import status

def pil_to_data_url(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return "data:image/png;base64," + encoded

def data_url_to_pil(value):
    if not value: raise ValueError("Empty image value")
    if value.startswith("data:"):
        _, b64 = value.split(",", 1)
        img_bytes = base64.b64decode(b64)
        return Image.open(BytesIO(img_bytes)).convert("RGB")
    return Image.open(os.path.join(settings.MEDIA_ROOT, value.lstrip('/'))).convert("RGB")

# ── CANNY FROM SCRATCH (VECTORIZED FOR SPEED) ─────────────────────────────────

class CannyScratch:
    @staticmethod
    def gaussian_kernel(size, sigma=1.0):
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        return g.astype(np.float32)

    @staticmethod
    def sobel_filters(img):
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
        
        Ix = cv2.filter2D(img, -1, Kx)
        Iy = cv2.filter2D(img, -1, Ky)
        
        G = np.hypot(Ix, Iy)
        G_max = G.max()
        if G_max == 0: G_max = 1e-5  # Prevent division by zero
        G = G / G_max * 255
        theta = np.arctan2(Iy, Ix)
        return G, theta

    @staticmethod
    def non_maximum_suppression(img, D):
        M, N = img.shape
        angle = D * 180. / np.pi
        angle[angle < 0] += 180
        
        # Pad image to handle borders safely and instantly
        img_pad = np.pad(img, 1, mode='constant')
        
        # Angle 0
        mask_0 = ((angle >= 0) & (angle < 22.5)) | ((angle >= 157.5) & (angle <= 180))
        q_0 = img_pad[1:M+1, 2:N+2]
        r_0 = img_pad[1:M+1, 0:N]
        
        # Angle 45
        mask_45 = (angle >= 22.5) & (angle < 67.5)
        q_45 = img_pad[2:M+2, 0:N]
        r_45 = img_pad[0:M, 2:N+2]
        
        # Angle 90
        mask_90 = (angle >= 67.5) & (angle < 112.5)
        q_90 = img_pad[2:M+2, 1:N+1]
        r_90 = img_pad[0:M, 1:N+1]
        
        # Angle 135
        mask_135 = (angle >= 112.5) & (angle < 157.5)
        q_135 = img_pad[0:M, 0:N]
        r_135 = img_pad[2:M+2, 2:N+2]
        
        # Build matrices
        q = np.zeros((M, N), dtype=np.float32)
        r = np.zeros((M, N), dtype=np.float32)
        
        q[mask_0] = q_0[mask_0]; r[mask_0] = r_0[mask_0]
        q[mask_45] = q_45[mask_45]; r[mask_45] = r_45[mask_45]
        q[mask_90] = q_90[mask_90]; r[mask_90] = r_90[mask_90]
        q[mask_135] = q_135[mask_135]; r[mask_135] = r_135[mask_135]
        
        # Keep if max
        Z = np.where((img >= q) & (img >= r), img, 0).astype(np.int32)
        return Z

    @staticmethod
    def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
        highThreshold = img.max() * highThresholdRatio
        lowThreshold = highThreshold * lowThresholdRatio
        
        res = np.zeros(img.shape, dtype=np.int32)
        weak = np.int32(25)
        strong = np.int32(255)
        
        strong_i, strong_j = np.where(img >= highThreshold)
        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
        
        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak
        
        return res, weak, strong

    @staticmethod
    def hysteresis(img, weak, strong=255):
        M, N = img.shape
        stack = []
        strong_i, strong_j = np.where(img == strong)
        for i, j in zip(strong_i, strong_j):
            stack.append((i, j))
            
        while stack:
            i, j = stack.pop()
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0: continue
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < M and 0 <= nj < N:
                        if img[ni, nj] == weak:
                            img[ni, nj] = strong
                            stack.append((ni, nj))
                            
        img[img != strong] = 0
        return img

    @classmethod
    def detect(cls, img_array, low_t, high_t):
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY).astype(np.float32)
        blur_kernel = cls.gaussian_kernel(5, sigma=1.4)
        blurred = cv2.filter2D(gray, -1, blur_kernel)
        mag, theta = cls.sobel_filters(blurred)
        nms = cls.non_maximum_suppression(mag, theta)
        
        high_ratio = high_t / 255.0
        low_ratio = low_t / 255.0
        thresh, weak, strong = cls.threshold(nms, low_ratio, high_ratio)
        
        final_edges = cls.hysteresis(thresh, weak, strong)
        final_edges = np.clip(final_edges, 0, 255).astype(np.uint8)
        return np.stack([final_edges, final_edges, final_edges], axis=2)

# ── HOUGH TRANSFORM FROM SCRATCH (VECTORIZED) ─────────────────────────────────

class HoughScratch:
    @staticmethod
    def detect_lines(img_array, threshold_votes=100):
        # 1. Get Edges (Using cv2.Canny here to quickly prep the image for the Hough math)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 2. Setup standard Hough Transform parameters
        height, width = edges.shape
        max_rho = int(np.ceil(np.sqrt(height**2 + width**2)))
        thetas = np.deg2rad(np.arange(-90, 90))
        rhos = np.arange(-max_rho, max_rho, 1)
        
        # 3. Vectorized Accumulator (Voting Matrix)
        accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int32)
        y_idxs, x_idxs = np.nonzero(edges) # Get all edge pixels instantly
        
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        
        # Fast voting: Calculate rho for all edge pixels simultaneously for each angle
        for i in range(len(thetas)):
            rho_vals = np.round(x_idxs * cos_t[i] + y_idxs * sin_t[i]).astype(int) + max_rho
            counts = np.bincount(rho_vals)
            accumulator[:len(counts), i] += counts
            
        # 4. Find peaks (Lines with more votes than our threshold)
        output_img = img_array.copy()
        rho_peaks, theta_peaks = np.where(accumulator > threshold_votes)
        
        # 5. Draw the mathematical lines back onto the image
        for r_idx, t_idx in zip(rho_peaks, theta_peaks):
            rho = rhos[r_idx]
            theta = thetas[t_idx]
            
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            
            # Extend the line across the image mathematically
            x1 = int(x0 + 2000 * (-b))
            y1 = int(y0 + 2000 * (a))
            x2 = int(x0 - 2000 * (-b))
            y2 = int(y0 - 2000 * (a))
            
            # Draw a thick blue line (BGR format in OpenCV, so RGB here is Red=(255,0,0) or Cyan=(0,255,255))
            cv2.line(output_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
        return output_img
    
class ActiveContourProcessor:
    @staticmethod
    def evolve_snake(img_array, init_radius=50):
        output_img = img_array.copy()
        dummy_chain_code = "0011223344556677"
        dummy_perimeter = 314.15
        dummy_area = 7853.98
        return output_img, dummy_chain_code, dummy_perimeter, dummy_area


# ── SESSION MANAGER ───────────────────────────────────────────────────────────

class SessionManager:
    DEFAULTS = {
        "img_history": [], "action_history": [], "low_history": [], "high_history": [], "hough_history": [],
        "img_original": None, "mode": "shapes", "last_action": "",
        "canny_low": 50, "canny_high": 150, "hough_thresh": 100,
        "chain_code": None, "perimeter": None, "area": None,
    }
    @classmethod
    def init_session(cls, session):
        for key, default in cls.DEFAULTS.items():
            if key not in session: session[key] = default
            
    @staticmethod
    def get_state(session):
        return {k: session.get(k, v) for k, v in SessionManager.DEFAULTS.items()}
    
    @staticmethod
    def save_state(session, state):
        for key, value in state.items(): session[key] = value
        session.modified = True


# ── API VIEWS ─────────────────────────────────────────────────────────────────

class StateView(APIView):
    def get(self, request):
        SessionManager.init_session(request.session)
        state = SessionManager.get_state(request.session)
        history = state.get("img_history", [])
        return Response({
            "mode": state["mode"], 
            "current_url": history[-1] if history else None,
            "original_url": state["img_original"], 
            "can_undo": len(history) > 1,
            "has_image": bool(history), 
            "last_action": state["last_action"],
            "canny_low": state["canny_low"], 
            "canny_high": state["canny_high"],
            "chain_code": state["chain_code"], 
            "perimeter": state["perimeter"], 
            "area": state["area"],
            "hough_thresh": state.get("hough_thresh", 100),
        })

class UploadView(APIView):
    parser_classes = [MultiPartParser, FormParser]
    def post(self, request):
        SessionManager.init_session(request.session)
        f = request.FILES.get("myfile")
        if not f: return Response({"error": "No file"}, status=400)

        pil_img = Image.open(f).convert("RGB")
        data_url = pil_to_data_url(pil_img)

        state = SessionManager.get_state(request.session)
        state.update({
            "img_history": [data_url], "action_history": [""],
            "low_history": [50], "high_history": [150], "hough_history": [100],
            "img_original": data_url, "last_action": "", 
            "canny_low": 50, "canny_high": 150, "hough_thresh": 100,
            "chain_code": None, "perimeter": None, "area": None
        })
        SessionManager.save_state(request.session, state)
        
        # FIX: Explicitly send empty strings/nulls back to React to clear the UI
        return Response({
            "current_url": data_url, 
            "original_url": data_url, 
            "has_image": True, 
            "can_undo": False,
            "last_action": "",
            "chain_code": None,
            "perimeter": None,
            "area": None
        })
class DetectShapesView(APIView):
    parser_classes = [JSONParser, MultiPartParser, FormParser]
    def post(self, request):
        state = SessionManager.get_state(request.session)
        if not state.get("img_original"):
            return Response({"error": "No original image found. Please upload the image again."}, status=400)
            
        shape_type = request.data.get("shape_type", "canny")
        low_t = int(request.data.get("canny_low", 50))
        high_t = int(request.data.get("canny_high", 150))
        h_thresh = int(request.data.get("hough_thresh", 100))
        
        try:
            arr = np.array(data_url_to_pil(state["img_original"]))
            action_text = ""
            
            if shape_type == "canny":
                out_arr = CannyScratch.detect(arr, low_t, high_t)
                action_text = f"Canny [{low_t}-{high_t}]"
            elif shape_type == "lines":
                out_arr = HoughScratch.detect_lines(arr, h_thresh)
                action_text = f"Lines [Votes > {h_thresh}]"
            else:
                out_arr = arr.copy() 
                action_text = f"Detected {shape_type}"
                
            new_url = pil_to_data_url(Image.fromarray(out_arr))
            
            state["img_history"].append(new_url)
            state["action_history"].append(action_text)
            state["low_history"].append(low_t)
            state["high_history"].append(high_t)
            state["hough_history"].append(h_thresh)
            
            state["last_action"] = state["action_history"][-1]
            state["canny_low"] = low_t
            state["canny_high"] = high_t
            state["hough_thresh"] = h_thresh
            SessionManager.save_state(request.session, state)
            
            return Response({
                "current_url": new_url, "original_url": state["img_original"],
                "last_action": state["last_action"], "can_undo": len(state["img_history"]) > 1
            })
        except Exception as e:
            import traceback
            traceback.print_exc() 
            return Response({"error": f"Backend Error: {str(e)}"}, status=500)
        
class ActiveContourView(APIView):
    parser_classes = [JSONParser, MultiPartParser, FormParser]
    def post(self, request):
        state = SessionManager.get_state(request.session)
        if not state.get("img_original"): return Response({"error": "No image found."}, status=400)
        try:
            arr = np.array(data_url_to_pil(state["img_original"]))
            out_arr, chain, perim, area = ActiveContourProcessor.evolve_snake(arr)
            new_url = pil_to_data_url(Image.fromarray(out_arr))
            
            state["img_history"].append(new_url)
            state["action_history"].append("Active Contour")
            state["low_history"].append(state["canny_low"]) # Carry over existing
            state["high_history"].append(state["canny_high"])
            
            state["last_action"] = state["action_history"][-1]
            state["chain_code"] = chain; state["perimeter"] = perim; state["area"] = area
            SessionManager.save_state(request.session, state)
            
            return Response({
                "current_url": new_url,
                "original_url": state["img_original"],
                "chain_code": chain, "perimeter": perim, "area": area,
                "last_action": state["last_action"],
                "can_undo": len(state["img_history"]) > 1
            })
        except Exception as e:
            return Response({"error": str(e)}, status=500)

class SwitchModeView(APIView):
    parser_classes = [JSONParser]
    def post(self, request):
        mode = request.data.get("mode", "shapes")
        state = SessionManager.get_state(request.session)
        
        # FIX: Reset the workspace automatically when changing tasks
        if state.get("img_original"):
            state["img_history"] = [state["img_original"]]
            state["action_history"] = [""]
            state["low_history"] = [50]
            state["high_history"] = [150]
            state["hough_history"] = [100]
            
            state["last_action"] = ""
            state["canny_low"] = 50
            state["canny_high"] = 150
            state["hough_thresh"] = 100
            state["chain_code"] = None
            state["perimeter"] = None
            state["area"] = None
            
        state["mode"] = mode
        SessionManager.save_state(request.session, state)
        
        # Return the clean slate back to React
        return Response({
            "mode": mode,
            "current_url": state.get("img_original"),
            "last_action": "",
            "can_undo": False,
            "chain_code": None,
            "perimeter": None,
            "area": None
        })

class UndoView(APIView):
    def post(self, request):
        state = SessionManager.get_state(request.session)
        if len(state["img_history"]) > 1:
            state["img_history"].pop()
            state["action_history"].pop()
            state["low_history"].pop()
            state["high_history"].pop()
            state["hough_history"].pop()
            
            state["last_action"] = state["action_history"][-1]
            state["canny_low"] = state["low_history"][-1]
            state["canny_high"] = state["high_history"][-1]
            state["hough_thresh"] = state["hough_history"][-1]
            SessionManager.save_state(request.session, state)
            
        return Response({
            "current_url": state["img_history"][-1] if state["img_history"] else None,
            "original_url": state["img_original"], "last_action": state["last_action"],
            "canny_low": state["canny_low"], "canny_high": state["canny_high"],
            "hough_thresh": state.get("hough_thresh", 100),
            "can_undo": len(state["img_history"]) > 1
        })

class ResetView(APIView):
    def post(self, request):
        state = SessionManager.get_state(request.session)
        if state["img_original"]:
            state["img_history"] = [state["img_original"]]
            state["action_history"] = [""]
            state["low_history"] = [50]
            state["high_history"] = [150]
            state["hough_history"] = [100]
            
            state["last_action"] = ""
            state["canny_low"] = 50
            state["canny_high"] = 150
            state["hough_thresh"] = 100
            state["chain_code"] = None; state["perimeter"] = None; state["area"] = None
            SessionManager.save_state(request.session, state)
            
        return Response({
            "current_url": state["img_original"], "original_url": state["img_original"],
            "last_action": state["last_action"], "canny_low": state["canny_low"],
            "canny_high": state["canny_high"], "hough_thresh": state["hough_thresh"],
            "can_undo": False
        })