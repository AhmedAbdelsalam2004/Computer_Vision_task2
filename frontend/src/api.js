import axios from 'axios';

// We just use /api now, Vite will automatically proxy it to port 8000!
const API_URL = '/api';

export const api = {
  getState: async () => {
    const res = await axios.get(`${API_URL}/state/`);
    return res.data;
  },
  upload: async (file) => {
    const formData = new FormData();
    formData.append("myfile", file);
    const res = await axios.post(`${API_URL}/upload/`, formData);
    return res.data;
  },
  detectShapes: async (data) => {
    const res = await axios.post(`${API_URL}/detect-shapes/`, data);
    return res.data;
  },
  activeContour: async (params = {}) => {
    const res = await axios.post(`${API_URL}/active-contour/`, params);
    return res.data;
  },
  undo: async () => {
    const res = await axios.post(`${API_URL}/undo/`);
    return res.data;
  },
  reset: async () => {
    const res = await axios.post(`${API_URL}/reset/`);
    return res.data;
  },
  switchMode: async (mode) => {
    const res = await axios.post(`${API_URL}/switch-mode/`, { mode });
    return res.data;
  }
};