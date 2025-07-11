import React, { useEffect, useState } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { fetchData, postData, selectError } from './demoSlice';

const Demo = () => {
  const dispatch = useDispatch();
  const { data, loading, error } = useSelector((state) => state.demo);
  const ogErr = useSelector(selectError);
  const [newPost, setNewPost] = useState('');

  useEffect(() => {
    dispatch(fetchData());
  }, [dispatch]);

  const handlePost = () => {
    if (newPost.trim() !== '') {
      dispatch(postData({ title: newPost, body: 'Demo post content' }));
      setNewPost('');
    }
  };

  return (
    <div className="min-h-screen p-6 flex flex-col items-center bg-gray-100">
      <h1 className="text-3xl font-bold mb-6">Demo Component</h1>
    </div>
  );
};

export default Demo;
