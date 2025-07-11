import { configureStore } from '@reduxjs/toolkit';
import demoReducer from '../features/demo/demoSlice';

export const store = configureStore({
  reducer: {
    demo: demoReducer,
  },
});

export default store;
