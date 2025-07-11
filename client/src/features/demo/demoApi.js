import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

// Fetch Data API
export const fetchDemoDataAPI = async () => {
  const response = await axios.get(`${API_BASE_URL}/posts`);
  return response.data;
};

// Post Data API
export const postDemoDataAPI = async (newData) => {
  const response = await axios.post('https://jsonplaceholder.typicode.com/posts', newData);
  return response.data;
};
