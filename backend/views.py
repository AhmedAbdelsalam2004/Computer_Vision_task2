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
    
# ── HOUGH CIRCLE TRANSFORM (100% PURE NUMPY / MATH, NO OPENCV) ─────────────────

class HoughCircleScratch:
    @staticmethod
    def rgb_to_gray(img_array):
        """Pure math RGB to Grayscale conversion (Luminosity method)"""
        return np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)

    @staticmethod
    def pure_numpy_edges(gray_img):
        """Finds edges using mathematical finite differences instead of cv2.Canny"""
        # Pad image by 1 pixel to handle borders safely
        padded = np.pad(gray_img, 1, mode='edge')
        
        # Calculate X and Y gradients by subtracting adjacent pixels
        dx = padded[1:-1, 2:] - padded[1:-1, :-2]
        dy = padded[2:, 1:-1] - padded[:-2, 1:-1]
        
        # Gradient magnitude (Pythagorean theorem)
        mag = np.hypot(dx, dy)
        
        # Normalize to 0-255
        mag = mag / (mag.max() + 1e-5) * 255
        
        # Keep only the strongest edges (Top 12% of gradients)
        thresh_val = np.percentile(mag, 88) 
        return np.where(mag > thresh_val)

    @staticmethod
    def draw_circle_numpy(img, cx, cy, r, color, thickness=2):
        """Draws a circle using pure coordinate math instead of cv2.circle"""
        h, w = img.shape[:2]
        y, x = np.ogrid[:h, :w]
        
        # Calculate distance of every pixel in the image to the center (cx, cy)
        dist = np.hypot(x - cx, y - cy)
        
        # Mask pixels that are exactly at distance 'r' (plus/minus thickness)
        outline_mask = (dist >= r - thickness) & (dist <= r + thickness)
        img[outline_mask] = color
        
        # Draw the center point (radius 4)
        center_mask = dist <= 4
        img[center_mask] = (255, 80, 80) # Blue dot
        
        return img

    @classmethod
    def detect_circles(cls, img_array, min_r=20, max_r=100, threshold_ratio=0.45, min_dist=None):
        h, w = img_array.shape[:2]
        
        # 1. Grayscale without CV
        gray = cls.rgb_to_gray(img_array)
        
        # 2. Edges without CV
        y_idxs, x_idxs = cls.pure_numpy_edges(gray)
        
        if len(y_idxs) == 0:
            return img_array.copy()
            
        # Performance safeguard: subsample edges if image is huge
        if len(y_idxs) > 15000:
            step = len(y_idxs) // 15000 + 1
            y_idxs = y_idxs[::step]
            x_idxs = x_idxs[::step]
            
        if min_dist is None:
            min_dist = max(min_r, 20)
            
        min_r = max(5, int(min_r))
        max_r = max(min_r + 1, int(max_r))
        r_step = max(1, (max_r - min_r) // 15)
        r_range = np.arange(min_r, max_r + 1, r_step)
        
        # 3. Build 3D Accumulator
        acc_3d = np.zeros((h, w, len(r_range)), dtype=np.int32)
        
        for ri, r in enumerate(r_range):
            num_angles = max(30, int(2 * np.pi * r))
            thetas = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
            
            cos_t = np.cos(thetas)
            sin_t = np.sin(thetas)
            
            # Vectorized broadcasting: calculate all voting coordinates simultaneously
            cx = np.round(x_idxs[:, None] + r * cos_t[None, :]).astype(np.int32)
            cy = np.round(y_idxs[:, None] + r * sin_t[None, :]).astype(np.int32)
            
            # Keep votes inside image boundaries
            valid = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
            np.add.at(acc_3d[:, :, ri], (cy[valid], cx[valid]), 1)
            
        # 4. Peak Finding (Greedy Non-Maximum Suppression)
        detected = []
        global_max = acc_3d.max()
        
        if global_max < 10:
            return img_array.copy()
            
        vote_threshold = max(15, int(global_max * threshold_ratio))
        
        for ri, r in enumerate(r_range):
            acc = acc_3d[:, :, ri]
            
            # Find all pixels that meet the vote threshold
            peaks_y, peaks_x = np.where(acc >= vote_threshold)
            scores = acc[peaks_y, peaks_x]
            
            # Sort them from highest votes to lowest
            sort_idx = np.argsort(scores)[::-1]
            peaks_y = peaks_y[sort_idx]
            peaks_x = peaks_x[sort_idx]
            scores = scores[sort_idx]
            
            for cy, cx, score in zip(peaks_y, peaks_x, scores):
                too_close = False
                
                # Check against already found circles based on pure distance
                for i, d in enumerate(detected):
                    dist = np.hypot(cx - d[0], cy - d[1])
                    if dist < min_dist:
                        too_close = True
                        # Overwrite if this new circle has more votes
                        if score > d[3]:
                            detected[i] = [cx, cy, r, score]
                        break
                        
                if not too_close:
                    detected.append([cx, cy, r, score])
                    
        # 5. Draw Circles without CV
        detected.sort(key=lambda x: x[3], reverse=True)
        output_img = img_array.copy()
        
        for cx, cy, r, score in detected[:30]:
            output_img = cls.draw_circle_numpy(output_img, cx, cy, r, (0, 255, 100), thickness=2)
            
        return output_img
    
class EllipseDetectorScratch:
    
    @staticmethod
    def draw_ellipse_numpy(img, cx, cy, a, b, angle_deg, color, thickness=2):
        """Colors pixels that fall on the mathematical boundary of a rotated ellipse"""
        h, w = img.shape[:2]
        y, x = np.ogrid[:h, :w]
        
        theta = np.radians(angle_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        dx = x - cx
        dy = y - cy
        
        # Rotate coordinates back to axis-aligned space
        rx = dx * cos_t + dy * sin_t
        ry = -dx * sin_t + dy * cos_t
        
        # Calculate distance in ellipse space (1.0 is exactly on the boundary)
        val = (rx / a)**2 + (ry / b)**2
        
        # Margin for line thickness
        margin = (thickness * 1.5) / min(a, b)
        
        mask = (val >= 1.0 - margin) & (val <= 1.0 + margin)
        img[mask] = color
        
        # Draw center point
        dist_c = np.hypot(dx, dy)
        img[dist_c <= 3] = (0, 200, 255) # Yellow dot
        
        return img
    
    
    @staticmethod
    def find_edge_groups(binary_img):
        """Groups touching edge pixels into contour arrays using BFS"""
        pts = np.argwhere(binary_img > 0)
        if len(pts) == 0: return []
        
        unvisited = set(map(tuple, pts))
        contours = []
        dirs = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        while unvisited:
            start = unvisited.pop()
            comp = [start]
            q = [start]
            while q:
                curr = q.pop()
                for d in dirs:
                    nxt = (curr[0] + d[0], curr[1] + d[1])
                    if nxt in unvisited:
                        unvisited.remove(nxt)
                        comp.append(nxt)
                        q.append(nxt)
                        
            # Convert from (y, x) to (x, y) coordinates
            contours.append(np.array([(p[1], p[0]) for p in comp]))
            
        return contours
    
    
    @staticmethod
    def math_morph_close(img_mask):
        """Dilation followed by Erosion using pure 3x3 array shifting"""
        p = np.pad(img_mask, 1, mode='edge')
        
        # Dilation (Max Pooling)
        dilated = np.maximum.reduce([
            p[0:-2, 0:-2], p[0:-2, 1:-1], p[0:-2, 2:],
            p[1:-1, 0:-2], p[1:-1, 1:-1], p[1:-1, 2:],
            p[2:, 0:-2],   p[2:, 1:-1],   p[2:, 2:]
        ])
        
        dp = np.pad(dilated, 1, mode='edge')
        
        # Erosion (Min Pooling)
        closed = np.minimum.reduce([
            dp[0:-2, 0:-2], dp[0:-2, 1:-1], dp[0:-2, 2:],
            dp[1:-1, 0:-2], dp[1:-1, 1:-1], dp[1:-1, 2:],
            dp[2:, 0:-2],   dp[2:, 1:-1],   dp[2:, 2:]
        ])
        return closed
    
    
    @staticmethod
    def _fit_conic(pts):
        """Algebraic least-squares conic fit in NORMALIZED space. Returns normed coeffs + scale info."""
        x = pts[:, 0].astype(np.float64)
        y = pts[:, 1].astype(np.float64)

        # Normalize for numerical stability
        mx, my = x.mean(), y.mean()
        scale = max(np.sqrt(((x - mx)**2 + (y - my)**2).mean()), 1e-6)
        xn = (x - mx) / scale
        yn = (y - my) / scale

        # Design matrix (N x 6): [x², xy, y², x, y, 1]
        D = np.column_stack([xn**2, xn*yn, yn**2, xn, yn, np.ones(len(xn))])
        S = D.T @ D  # Scatter matrix

        # Constraint matrix for ellipse: 4AC - B² = 1
        C_mat = np.zeros((6, 6))
        C_mat[0, 2] = 2.0; C_mat[2, 0] = 2.0; C_mat[1, 1] = -1.0

        try:
            # Add small regularization for numerical stability
            eigvals, eigvecs = np.linalg.eig(np.linalg.solve(S + 1e-10 * np.eye(6), C_mat))
        except np.linalg.LinAlgError:
            return None, None, None

        eigvals = eigvals.real
        eigvecs = eigvecs.real

        # Pick the eigenvector with the smallest POSITIVE eigenvalue
        pos_mask = (eigvals > 1e-12) & np.isfinite(eigvals)
        if not pos_mask.any():
            return None, None, None

        best_idx = np.where(pos_mask)[0][np.argmin(eigvals[pos_mask])]
        a_norm = eigvecs[:, best_idx]

        # Validate it really describes an ellipse: 4AC - B² > 0
        A_, B_, C_, D_, E_, F_ = a_norm
        if (B_**2 - 4 * A_ * C_) >= 0:
            return None, None, None

        return a_norm, (mx, my), scale

    @staticmethod
    def _conic_to_ellipse_normalized(a_norm, center_shift, scale):
        """
        Convert normalized-space conic coefficients back to image space
        and extract ellipse geometry.
        """
        A_, B_, C_, D_, E_, F_ = a_norm.astype(np.float64)
        mx, my = center_shift
        s = scale

        # De-normalize the coefficients to original pixel space
        # Substituting x = (X - mx)/s, y = (Y - my)/s back:
        A = A_ / s**2
        B = B_ / s**2
        C = C_ / s**2
        D = (D_ - 2 * A_ * mx / s - B_ * my / s) / s
        E = (E_ - B_ * mx / s - 2 * C_ * my / s) / s
        F = (A_ * (mx/s)**2 + B_ * (mx/s) * (my/s) + C_ * (my/s)**2
             - D_ * (mx/s) - E_ * (my/s) + F_)

        # Verify ellipse discriminant
        if B**2 - 4*A*C >= 0:
            return None

        # Solve for center: dF/dx = 2Ax + By + D = 0, dF/dy = Bx + 2Cy + E = 0
        try:
            center = np.linalg.solve([[2*A, B], [B, 2*C]], [-D, -E])
        except np.linalg.LinAlgError:
            return None
        cx_e, cy_e = center

        # Value of conic at center (used to derive semi-axes)
        Fp = A*cx_e**2 + B*cx_e*cy_e + C*cy_e**2 + D*cx_e + E*cy_e + F
        if abs(Fp) < 1e-12:
            return None

        # Semi-axes and orientation from eigendecomposition of M = [[A, B/2], [B/2, C]]
        # eigh returns eigenvalues in ASCENDING order; smaller eigenvalue → larger semi-axis
        eig_vals, eig_vecs = np.linalg.eigh([[A, B/2.0], [B/2.0, C]])
        vals = -Fp / eig_vals
        if np.any(vals <= 0) or np.any(~np.isfinite(vals)):
            return None
        semi = np.sqrt(vals)  # semi[0] ↔ eig_vals[0] (smaller), semi[1] ↔ eig_vals[1] (larger)

        # Identify which index is the MAJOR axis (larger semi-axis)
        major_idx = int(np.argmax(semi))   # index of larger semi-axis
        minor_idx = 1 - major_idx

        a_semi = float(semi[major_idx])    # semi-major
        b_semi = float(semi[minor_idx])    # semi-minor

        # Angle of the major axis = direction of the corresponding eigenvector
        # eig_vecs[:, idx] is the eigenvector for eig_vals[idx]
        major_vec = eig_vecs[:, major_idx]
        angle_rad = np.arctan2(major_vec[1], major_vec[0])
        angle_deg = float(np.degrees(angle_rad))

        return (cx_e, cy_e), (a_semi, b_semi), angle_deg

    @classmethod
    def detect_ellipses(cls, img_array, min_area=200):
        h, w = img_array.shape[:2]
        
        # 1. Get Edges (Using the new pure math edge function from the Circle class)
        gray = HoughCircleScratch.rgb_to_gray(img_array)
        y_idx, x_idx = HoughCircleScratch.pure_numpy_edges(gray)
        
        binary_edges = np.zeros((h, w), dtype=np.uint8)
        binary_edges[y_idx, x_idx] = 1
        
        # 2. Close Gaps (No cv2.morphologyEx)
        closed_edges = cls.math_morph_close(binary_edges)
        
        # 3. Find Contours (No cv2.findContours)
        contours = cls.find_edge_groups(closed_edges)
        
        output_img = img_array.copy()
        drawn_centers = []

        for pts in contours:
            if len(pts) < 15: # Skip tiny fragments
                continue
                
            # 4. Filter by Bounding Box Area (No cv2.contourArea)
            min_x, min_y = np.min(pts, axis=0)
            max_x, max_y = np.max(pts, axis=0)
            bbox_area = (max_x - min_x) * (max_y - min_y)
            if bbox_area < min_area:
                continue

            pts_float = pts.astype(np.float64)

            # Subsample large contours for fitting speed
            if len(pts_float) > 300:
                idx = np.linspace(0, len(pts_float) - 1, 300, dtype=int)
                pts_float = pts_float[idx]

            a_norm, center_shift, scale = cls._fit_conic(pts_float)
            if a_norm is None: continue

            result = cls._conic_to_ellipse_normalized(a_norm, center_shift, scale)
            if result is None: continue

            (ecx, ecy), (a, b), angle = result

            # Sanity checks
            if a < 3 or b < 3 or a > 2 * max(h, w) or b / a < 0.05:
                continue
            if ecx < -a or ecx > w + a or ecy < -b or ecy > h + b:
                continue

            # Check for duplicates
            if any(np.hypot(ecx - dc[0], ecy - dc[1]) < max(a, b) * 0.5 for dc in drawn_centers):
                continue

            # 5. Draw the result (No cv2.ellipse)
            output_img = cls.draw_ellipse_numpy(
                output_img, 
                cx=ecx, cy=ecy, a=a, b=b, 
                angle_deg=angle, color=(255, 180, 0)
            )
            drawn_centers.append((ecx, ecy))

        return output_img


# ── ACTIVE CONTOUR (SNAKE) FROM SCRATCH ───────────────────────────────────────

class ActiveContourProcessor:
    @staticmethod
    def compute_external_energy(gray):
        """External energy = negative of gradient magnitude (attract snake to edges)"""
        gray_f = gray.astype(np.float32)
        gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        # Blur to create wider attraction basin
        mag = cv2.GaussianBlur(mag, (21, 21), 5)
        # Normalise to [0, 1]
        mx = mag.max()
        if mx > 0:
            mag = mag / mx
        return -mag  # snake moves toward negative (strong edges)

    @staticmethod
    def build_snake_matrix(n, alpha, beta, gamma):
        """Build the pentadiagonal regularisation matrix A for linearised snake update."""
        # Internal energy: alpha * (elasticity) + beta * (curvature stiffness)
        a = beta
        b = -(alpha + 4 * beta)
        c = 2 * alpha + 6 * beta
        row = np.zeros(n)
        row[0] = c
        row[1] = b;  row[-1] = b
        row[2] = a;  row[-2] = a
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                A[i, j] = row[(j - i) % n]
        return A

    @staticmethod
    def _init_from_points(init_points, w, h, n_points):
        """Resample user-drawn path (normalized 0-1 coords) to n_points snake."""
        pts = np.array(init_points, dtype=np.float64)
        pts[:, 0] *= w;  pts[:, 1] *= h   # scale to image pixels
        if not np.allclose(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0]])   # close the path
        diffs = np.diff(pts, axis=0)
        dists = np.maximum(np.sqrt((diffs**2).sum(axis=1)), 1e-9)
        cumlen = np.concatenate([[0], np.cumsum(dists)])
        t_new = np.linspace(0, cumlen[-1], n_points, endpoint=False)
        sx = np.interp(t_new, cumlen, pts[:, 0])
        sy = np.interp(t_new, cumlen, pts[:, 1])
        return sx, sy

    @classmethod
    def evolve_snake(cls, img_array, init_points=None, alpha=0.1, beta=0.1, gamma=0.5, iterations=200, n_points=120):
        """
        alpha      : elasticity (tension)
        beta       : stiffness (rigidity)
        gamma      : step size
        init_points: list of [nx, ny] in [0,1] range (user-drawn path) or None → default circle
        """
        h, w = img_array.shape[:2]
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # ── Initialize snake from user path or default circle ─────────────────
        if init_points and len(init_points) >= 3:
            snake_x, snake_y = cls._init_from_points(init_points, w, h, n_points)
        else:
            cx, cy = w // 2, h // 2
            r = min(cx, cy) * 0.6
            t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
            snake_x = cx + r * np.cos(t)
            snake_y = cy + r * np.sin(t)

        # External energy map and its gradient
        ext_energy = cls.compute_external_energy(gray)
        # Gradient of external energy (force field)
        fx = cv2.Sobel(ext_energy.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        fy = cv2.Sobel(ext_energy.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)

        # Build internal energy matrix and pre-invert: (A + gamma*I)^-1
        A = cls.build_snake_matrix(n_points, alpha, beta, gamma)
        inv_matrix = np.linalg.inv(A + gamma * np.eye(n_points))

        for _ in range(iterations):
            # Sample external forces at current snake positions (bilinear)
            xi = np.clip(snake_x.astype(int), 0, w - 1)
            yi = np.clip(snake_y.astype(int), 0, h - 1)
            Fx = -fx[yi, xi]
            Fy = -fy[yi, xi]

            # Update: x_{t+1} = inv_matrix @ (gamma * x_t + Fx)
            snake_x = inv_matrix @ (gamma * snake_x + Fx)
            snake_y = inv_matrix @ (gamma * snake_y + Fy)

            # Clamp inside image
            snake_x = np.clip(snake_x, 0, w - 1)
            snake_y = np.clip(snake_y, 0, h - 1)

        # ── Draw final snake contour ──────────────────────────────────────────
        output_img = img_array.copy()
        pts = np.stack([snake_x, snake_y], axis=1).astype(np.int32)
        cv2.polylines(output_img, [pts], isClosed=True, color=(0, 255, 180), thickness=2)
        # Mark snake points
        for px, py in pts[::4]:
            cv2.circle(output_img, (px, py), 3, (255, 80, 80), -1)

        # ── Chain Code (8-connectivity Freeman) ──────────────────────────────
        dx_dirs = [1, 1, 0, -1, -1, -1,  0,  1]
        dy_dirs = [0, 1, 1,  1,  0, -1, -1, -1]
        chain_list = []
        for i in range(len(pts)):
            p1 = pts[i]
            p2 = pts[(i + 1) % len(pts)]
            ddx = int(np.sign(p2[0] - p1[0]))
            ddy = int(np.sign(p2[1] - p1[1]))
            # Find closest direction
            best_dir = 0
            best_dot = -99
            for d in range(8):
                dot = ddx * dx_dirs[d] + ddy * dy_dirs[d]
                if dot > best_dot:
                    best_dot = dot
                    best_dir = d
            chain_list.append(str(best_dir))
        chain_code = "".join(chain_list)

        # ── Perimeter (sum of Euclidean segment lengths) ───────────────────────
        perimeter = 0.0
        for i in range(len(pts)):
            p1 = pts[i].astype(float)
            p2 = pts[(i + 1) % len(pts)].astype(float)
            perimeter += np.linalg.norm(p2 - p1)

        # ── Area (Shoelace formula) ────────────────────────────────────────────
        xs = pts[:, 0].astype(float)
        ys = pts[:, 1].astype(float)
        area = 0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(np.roll(xs, -1), ys))

        return output_img, chain_code, round(perimeter, 2), round(area, 2)


# ── SESSION MANAGER ───────────────────────────────────────────────────────────

class SessionManager:
    DEFAULTS = {
        "img_history": [], "action_history": [], "low_history": [], "high_history": [],
        "hough_history": [], "circle_history": [], "ellipse_history": [],
        "img_original": None, "mode": "shapes", "last_action": "",
        "canny_low": 50, "canny_high": 150, "hough_thresh": 100,
        "circle_min_r": 20, "circle_max_r": 100, "circle_thresh": 45,
        "ellipse_min_area": 200,
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
            "circle_min_r": state.get("circle_min_r", 20),
            "circle_max_r": state.get("circle_max_r", 100),
            "circle_thresh": state.get("circle_thresh", 45),
            "ellipse_min_area": state.get("ellipse_min_area", 200),
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
            "circle_history": [(20, 100, 45)], "ellipse_history": [200],
            "img_original": data_url, "last_action": "",
            "canny_low": 50, "canny_high": 150, "hough_thresh": 100,
            "circle_min_r": 20, "circle_max_r": 100, "circle_thresh": 45,
            "ellipse_min_area": 200,
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
        low_t  = int(request.data.get("canny_low", 50))
        high_t = int(request.data.get("canny_high", 150))
        h_thresh = int(request.data.get("hough_thresh", 100))
        c_min_r  = int(request.data.get("circle_min_r", 20))
        c_max_r  = int(request.data.get("circle_max_r", 100))
        c_thresh = int(request.data.get("circle_thresh", 45))
        e_min_area = int(request.data.get("ellipse_min_area", 200))

        try:
            arr = np.array(data_url_to_pil(state["img_original"]))
            action_text = ""

            if shape_type == "canny":
                out_arr = CannyScratch.detect(arr, low_t, high_t)
                action_text = f"Canny [{low_t}-{high_t}]"
            elif shape_type == "lines":
                out_arr = HoughScratch.detect_lines(arr, h_thresh)
                action_text = f"Lines [Votes>{h_thresh}]"
            elif shape_type == "circles":
                out_arr = HoughCircleScratch.detect_circles(
                    arr, min_r=c_min_r, max_r=c_max_r,
                    threshold_ratio=c_thresh / 100.0
                )
                action_text = f"Circles [r={c_min_r}-{c_max_r}, t={c_thresh}%]"
            elif shape_type == "ellipses":
                out_arr = EllipseDetectorScratch.detect_ellipses(arr, min_area=e_min_area)
                action_text = f"Ellipses [area>{e_min_area}]"
            else:
                out_arr = arr.copy()
                action_text = f"Detected {shape_type}"

            new_url = pil_to_data_url(Image.fromarray(out_arr))

            state["img_history"].append(new_url)
            state["action_history"].append(action_text)
            state["low_history"].append(low_t)
            state["high_history"].append(high_t)
            state["hough_history"].append(h_thresh)
            state["circle_history"].append((c_min_r, c_max_r, c_thresh))
            state["ellipse_history"].append(e_min_area)

            state["last_action"]   = action_text
            state["canny_low"]     = low_t
            state["canny_high"]    = high_t
            state["hough_thresh"]  = h_thresh
            state["circle_min_r"]  = c_min_r
            state["circle_max_r"]  = c_max_r
            state["circle_thresh"] = c_thresh
            state["ellipse_min_area"] = e_min_area
            SessionManager.save_state(request.session, state)

            return Response({
                "current_url": new_url,
                "original_url": state["img_original"],
                "last_action": state["last_action"],
                "can_undo": len(state["img_history"]) > 1,
                "circle_min_r": c_min_r, "circle_max_r": c_max_r,
                "circle_thresh": c_thresh, "ellipse_min_area": e_min_area,
            })
        except Exception as e:
            traceback.print_exc()
            return Response({"error": f"Backend Error: {str(e)}"}, status=500)
        
class ActiveContourView(APIView):
    parser_classes = [JSONParser, MultiPartParser, FormParser]
    def post(self, request):
        state = SessionManager.get_state(request.session)
        if not state.get("img_original"): return Response({"error": "No image found."}, status=400)
        try:
            alpha      = float(request.data.get("alpha", 0.1))
            beta       = float(request.data.get("beta",  0.1))
            gamma      = float(request.data.get("gamma", 0.5))
            iterations = int(request.data.get("iterations", 200))

            init_points = request.data.get("init_points", None)  # list of [nx,ny] in [0,1]

            arr = np.array(data_url_to_pil(state["img_original"]))
            out_arr, chain, perim, area = ActiveContourProcessor.evolve_snake(
                arr, init_points=init_points,
                alpha=alpha, beta=beta, gamma=gamma, iterations=iterations
            )
            new_url = pil_to_data_url(Image.fromarray(out_arr))

            state["img_history"].append(new_url)
            state["action_history"].append("Active Contour")
            state["low_history"].append(state["canny_low"])
            state["high_history"].append(state["canny_high"])
            state["hough_history"].append(state.get("hough_thresh", 100))
            if "circle_history" not in state: state["circle_history"] = []
            state["circle_history"].append((state.get("circle_min_r", 20), state.get("circle_max_r", 100), state.get("circle_thresh", 45)))
            if "ellipse_history" not in state: state["ellipse_history"] = []
            state["ellipse_history"].append(state.get("ellipse_min_area", 200))

            state["last_action"] = "Active Contour"
            state["chain_code"] = chain
            state["perimeter"]  = perim
            state["area"]       = area
            SessionManager.save_state(request.session, state)

            return Response({
                "current_url": new_url,
                "original_url": state["img_original"],
                "chain_code": chain, "perimeter": perim, "area": area,
                "last_action": state["last_action"],
                "can_undo": len(state["img_history"]) > 1
            })
        except Exception as e:
            traceback.print_exc()
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
            if state.get("circle_history") and len(state["circle_history"]) > 1:
                state["circle_history"].pop()
            if state.get("ellipse_history") and len(state["ellipse_history"]) > 1:
                state["ellipse_history"].pop()

            state["last_action"]  = state["action_history"][-1]
            state["canny_low"]    = state["low_history"][-1]
            state["canny_high"]   = state["high_history"][-1]
            state["hough_thresh"] = state["hough_history"][-1]
            c_hist = state.get("circle_history", [(20, 100, 45)])
            state["circle_min_r"], state["circle_max_r"], state["circle_thresh"] = c_hist[-1] if c_hist else (20, 100, 45)
            e_hist = state.get("ellipse_history", [200])
            state["ellipse_min_area"] = e_hist[-1] if e_hist else 200
            SessionManager.save_state(request.session, state)

        return Response({
            "current_url": state["img_history"][-1] if state["img_history"] else None,
            "original_url": state["img_original"],
            "last_action": state["last_action"],
            "canny_low": state["canny_low"], "canny_high": state["canny_high"],
            "hough_thresh": state.get("hough_thresh", 100),
            "circle_min_r": state.get("circle_min_r", 20),
            "circle_max_r": state.get("circle_max_r", 100),
            "circle_thresh": state.get("circle_thresh", 45),
            "ellipse_min_area": state.get("ellipse_min_area", 200),
            "can_undo": len(state["img_history"]) > 1
        })

class ResetView(APIView):
    def post(self, request):
        state = SessionManager.get_state(request.session)
        if state["img_original"]:
            state["img_history"]    = [state["img_original"]]
            state["action_history"] = [""]
            state["low_history"]    = [50]
            state["high_history"]   = [150]
            state["hough_history"]  = [100]
            state["circle_history"] = [(20, 100, 45)]
            state["ellipse_history"]= [200]

            state["last_action"]      = ""
            state["canny_low"]        = 50
            state["canny_high"]       = 150
            state["hough_thresh"]     = 100
            state["circle_min_r"]     = 20
            state["circle_max_r"]     = 100
            state["circle_thresh"]    = 45
            state["ellipse_min_area"] = 200
            state["chain_code"]       = None
            state["perimeter"]        = None
            state["area"]             = None
            SessionManager.save_state(request.session, state)

        return Response({
            "current_url": state["img_original"],
            "original_url": state["img_original"],
            "last_action": state["last_action"],
            "canny_low": state["canny_low"], "canny_high": state["canny_high"],
            "hough_thresh": state["hough_thresh"],
            "circle_min_r": state["circle_min_r"],
            "circle_max_r": state["circle_max_r"],
            "circle_thresh": state["circle_thresh"],
            "ellipse_min_area": state["ellipse_min_area"],
            "can_undo": False
        })