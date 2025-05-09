import React, { useState } from 'react'; 
import { useSelector, useDispatch } from 'react-redux';
import { deleteQuiz } from '../features/quiz/quizSlice';
import { useNavigate } from 'react-router-dom';

const QuizCard = () => {
  const quizes = useSelector((state) => state.quiz.quizes);
  const [searchTerm, setSearchTerm] = useState('');
  const dispatch = useDispatch();
  const navigate = useNavigate();

  const getDifficultyGradient = (difficulty = '') => {
    switch (difficulty.toLowerCase()) {
      case 'easy':
        return 'bg-gradient-to-r from-green-400 via-green-500 to-green-600'; 
      case 'medium':
        return 'bg-gradient-to-r from-yellow-400 via-yellow-500 to-yellow-600'; 
      case 'hard':
        return 'bg-gradient-to-r from-red-400 via-red-500 to-red-600'; 
      default:
        return 'bg-gradient-to-r from-gray-400 via-gray-500 to-gray-600'; 
    }
  };

  const deleteHandle = (id) => {
    dispatch(deleteQuiz(id));
  };

  const updateHandle = (id) => {
    navigate(`/quiz/${id}`);
  };

  const attemptHandle = (id) => {
    navigate(`/quiz/attempt/${id}`);
  };

  // Filter quizzes by title based on search term
  const filteredQuizzes = quizes.filter((quiz) =>
    quiz.title.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="p-4 max-w-[1080px] mx-auto mt-[2rem]">
      {/* Search Bar */}
      <div className="mb-6">
        <input
          type="text"
          placeholder="Search by title..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="w-[30%] p-2 pl-4 rounded-[2rem] border-2 border-gray-300 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredQuizzes.length === 0 ? (
          <p>No quizzes found.</p>
        ) : (
          filteredQuizzes.map((quiz) => (
            <div key={quiz.id} className={`border-2 p-4 rounded-lg shadow-md ${getDifficultyGradient(quiz.description)} transition duration-300 hover:scale-[0.9]`}>
              <div className='h-[50px] w-[50px] border-2 p-2 rounded-full border-white'>
                <img
                  src={`images/${quiz.description}.png`}
                  alt={`Difficulty: ${quiz.description}`}
                  className="w-full h-full"
                />
              </div>
              <h2 className="text-xl font-semibold text-white mt-1">{quiz.title}</h2>
              <p className="italic text-white mt-2">Difficulty: {quiz.description}</p>
              <div className='flex flex-col gap-2 mt-4 mx-8'>
                <button className="bg-gradient-to-r from-[#56CCF2] via-[#2F80ED] to-[#56CCF2] bg-[length:200%_auto] hover:bg-[position:right_center] text-white uppercase font-semibold px-[1.5rem] py-[0.3rem] rounded-lg shadow transition-all duration-500" onClick={() => updateHandle(quiz.id)}>
                  Update
                </button>
                <button className="bg-gradient-to-r from-[#56CCF2] via-[#2F80ED] to-[#56CCF2] bg-[length:200%_auto] hover:bg-[position:right_center] text-white uppercase font-semibold px-[1.5rem] py-[0.3rem] rounded-lg shadow transition-all duration-500" onClick={() => attemptHandle(quiz.id)}>
                  Attempt
                </button>
                <button className="bg-gradient-to-r from-[#56CCF2] via-[#2F80ED] to-[#56CCF2] bg-[length:200%_auto] hover:bg-[position:right_center] text-white uppercase font-semibold px-[1.5rem] py-[0.3rem] rounded-lg shadow transition-all duration-500" onClick={() => deleteHandle(quiz.id)}>
                  Delete
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default QuizCard;
