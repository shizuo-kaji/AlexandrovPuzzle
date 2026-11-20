import { exportDesign } from './export-design.js';

self.onmessage = async ({ data }) => {
  try {
    self.postMessage(await exportDesign(data));
  } catch (error) {
    self.postMessage({ error: error.name, message: error.message });
  }
};
