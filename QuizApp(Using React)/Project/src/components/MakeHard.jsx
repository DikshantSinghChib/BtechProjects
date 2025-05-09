import React, { useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { addQuiz } from '../features/quiz/quizSlice';
import { v4 as uuidv4 } from 'uuid';
import { useParams, useNavigate } from 'react-router-dom';

const MakeHard = () => {
  const [title, setTitle] = useState('');
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const [quiz, setQuiz] = useState([
    {
      question: '',
      options: ['', '', '', ''],
      answer: ''
    }
  ]);

  const handleQuizChange = (index, field, value) => {
    const updatedQuiz = [...quiz];
    updatedQuiz[index][field] = value;
    setQuiz(updatedQuiz);
  };

  const handleOptionChange = (qIndex, optIndex, value) => {
    const updatedQuiz = [...quiz];
    updatedQuiz[qIndex].options[optIndex] = value;
    setQuiz(updatedQuiz);
  };

  const addQuestion = () => {
    if (quiz.length < 25) {
      setQuiz([
        ...quiz,
        {
          question: '',
          options: ['', '', '', ''],
          answer: ''
        }
      ]);
    } else {
      alert('Maximum of 5 questions allowed.');
    }
  };
  function createQuiz() {
      const dataQuiz = {
        id: uuidv4(),
        title,
        quiz,
        description: "hard"
      };
  
      dispatch(addQuiz(dataQuiz));
      navigate('/');
  };
  return (
    <div className="mx-[2rem] my-[1rem] border-4 p-4 rounded-lg">
      <h2 className="font-bold text-2xl pb-4">Making Quiz</h2>
      <form className="flex flex-col gap-6 justify-center">
        {/* Title */}
        <div className="flex flex-row gap-4 items-center">
          <label className="text-[1.2rem]">Title:</label>
          <input
            type="text"
            name="title"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Input the title"
            className="w-1/2 p-1 border-2 rounded-lg"
          />
        </div>

        {/* Quiz Questions */}
        {quiz.map((q, qIndex) => (
          <div key={qIndex} className="flex flex-col gap-2">
            <h2 className="font-bold text-lg">Question {qIndex + 1}:</h2>
            <input
              type="text"
              name={`question${qIndex}`}
              placeholder="Enter question"
              className="w-1/2 p-1 border-2 rounded-lg"
              value={q.question}
              onChange={(e) =>
                handleQuizChange(qIndex, 'question', e.target.value)
              }
            />

            {/* Options */}
            <div className="ml-10 mt-2">
              {q.options.map((opt, optIndex) => (
                <div key={optIndex} className="mb-4">
                  <label className="font-bold">Option {optIndex + 1}:</label>
                  <input
                    type="text"
                    placeholder="Enter option"
                    className="w-1/2 ml-2 p-1 border-2 rounded-lg"
                    value={opt}
                    onChange={(e) =>
                      handleOptionChange(qIndex, optIndex, e.target.value)
                    }
                  />
                </div>
              ))}

              {/* Answer */}
              <label className="font-bold">Answer:</label>
              <input
                type="text"
                placeholder="Enter Answer"
                className="w-1/2 ml-2 p-1 border-2 rounded-lg"
                value={q.answer}
                onChange={(e) =>
                  handleQuizChange(qIndex, 'answer', e.target.value)
                }
              />
            </div>
          </div>
        ))}

        {/* Add Question Button */}
        <button
          type="button"
          onClick={addQuestion}
          disabled={quiz.length >= 25}
          className={`mt-4 w-fit px-4 py-2 text-white rounded-lg ${
            quiz.length >= 25
              ? 'bg-gray-400 cursor-not-allowed'
              : 'bg-blue-500 hover:bg-blue-600'
          }`}
        >
          + Add Question
        </button>
        <button
          type="button"
          onClick={createQuiz}
          className="w-fit px-4 py-2 text-white bg-green-500 rounded-lg hover:bg-green-600"
        >
          Create Quiz
        </button>
      </form>

    </div>
  );
};

export default MakeHard
