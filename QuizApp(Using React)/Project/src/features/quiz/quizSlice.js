import { createSlice } from '@reduxjs/toolkit'
import toast from 'react-hot-toast';

const initialState = {
  quizes: JSON.parse(localStorage.getItem('quizes')) || [],
};

export const quizSlice = createSlice({
  name: 'quiz',
  initialState,
  reducers: {
    addQuiz: (state,action) => {
        const data = action.payload;
        state.quizes.push(data);
        localStorage.setItem("quizes", JSON.stringify(state.quizes));
        toast.success("Quiz Created");
    },
    deleteQuiz: (state,action) => {
      const id=action.payload;
      const index = state.quizes.findIndex((item)=> item.id==id);
      if(index>=0)
      {
        state.quizes.splice(index, 1);
        localStorage.setItem("quizes", JSON.stringify(state.quizes));
      }
      toast.success("Quiz Deleted");
    },
    updateQuiz: (state, action) => {
      const data = action.payload;
      const index = state.quizes.findIndex((item)=> item.id===data.id);

      if(index>=0)
      {
        state.quizes[index]=data;
        localStorage.setItem("quizes", JSON.stringify(state.quizes));
      }
      toast.success("Update Quiz");
    },
  },
})

// Action creators are generated for each case reducer function
export const { addQuiz, deleteQuiz, updateQuiz } = quizSlice.actions

export default quizSlice.reducer