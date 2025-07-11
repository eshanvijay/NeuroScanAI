import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { fetchDemoDataAPI, postDemoDataAPI } from './demoAPI';

const initialState = {
  data: [],
  loading: false,
  error: null,
};

// Async Thunk for Fetching Data
export const fetchData = createAsyncThunk('demo/fetchData', async (_, { rejectWithValue }) => {
  try {
    const response = await fetchDemoDataAPI();
    return response;
  } catch (error) {
    return rejectWithValue(error.message);
  }
});

// Async Thunk for Posting Data
export const postData = createAsyncThunk('demo/postData', async (newData, { rejectWithValue }) => {
  try {
    const response = await postDemoDataAPI(newData);
    return response;
  } catch (error) {
    return rejectWithValue(error.message);
  }
});

const demoSlice = createSlice({
  name: 'demo',
  initialState,
  reducers: {},

  extraReducers: (builder) => {
    builder
      // Handle Fetch Data
      .addCase(fetchData.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchData.fulfilled, (state, action) => {
        state.loading = false;
        state.data = action.payload;
      })
      .addCase(fetchData.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload;
      })

      // Handle Post Data
      .addCase(postData.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(postData.fulfilled, (state, action) => {
        state.loading = false;
        state.data.push(action.payload);
      })
      .addCase(postData.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload;
      });
  },
});

export const selectCount = (state) => state.demo.value;
export const selectStatus = (state) => state.demo.status;
export const selectError = (state) => state.demo.error;

export default demoSlice.reducer;
