import React, { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { useParams, useNavigate } from "react-router-dom";
import { updateQuiz } from '../features/quiz/quizSlice';

const Update = () => {
  const quizes = useSelector((state) => state.quiz.quizes);
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const { id } = useParams();

  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [quiz, setQuiz] = useState([]);

  useEffect(() => {
    const foundQuiz = quizes.find((q) => q.id === id);
    if (foundQuiz) {
      setTitle(foundQuiz.title);
      setDescription(foundQuiz.description || '');
      setQuiz(foundQuiz.quiz);
    }
  }, [id, quizes]);

  const handleQuizChange = (index, field, value) => {
    const updatedQuiz = quiz.map((q, i) =>
      i === index ? { ...q, [field]: value } : q
    );
    setQuiz(updatedQuiz);
  };

  const handleOptionChange = (qIndex, optIndex, value) => {
    const updatedQuiz = quiz.map((q, i) => {
      if (i !== qIndex) return q;
      const newOptions = [...q.options];
      newOptions[optIndex] = value;
      return { ...q, options: newOptions };
    });
    setQuiz(updatedQuiz);
  };

  const handleUpdate = () => {
    const updatedData = {
      id,
      title,
      quiz,
      description: description.toLowerCase(),
    };
    dispatch(updateQuiz(updatedData));
    navigate('/');
  };

  return (
    <div className="mx-[2rem] my-[1rem] border-4 p-4 rounded-lg">
      <h2 className="font-bold text-2xl pb-4">Update Quiz</h2>
      <form className="flex flex-col gap-6 justify-center">
        <div className="flex flex-row gap-4 items-center">
          <label className="text-[1.2rem]">Title:</label>
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            className="w-1/2 p-1 border-2 rounded-lg"
          />
        </div>

        <div className="flex flex-row gap-4 items-center">
          <label className="text-[1.2rem]">Difficulty:</label>
          <input
            type="text"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            className="w-1/2 p-1 border-2 rounded-lg"
            placeholder="e.g. easy, medium, hard"
          />
        </div>

        {quiz.map((q, qIndex) => (
          <div key={qIndex} className="flex flex-col gap-2">
            <h2 className="font-bold text-lg">Question {qIndex + 1}:</h2>
            <input
              type="text"
              value={q.question}
              onChange={(e) =>
                handleQuizChange(qIndex, 'question', e.target.value)
              }
              className="w-1/2 p-1 border-2 rounded-lg"
            />
            <div className="ml-10 mt-2">
              {q.options.map((opt, optIndex) => (
                <div key={optIndex} className="mb-4">
                  <label className="font-bold">Option {optIndex + 1}:</label>
                  <input
                    type="text"
                    value={opt}
                    onChange={(e) =>
                      handleOptionChange(qIndex, optIndex, e.target.value)
                    }
                    className="w-1/2 ml-2 p-1 border-2 rounded-lg"
                  />
                </div>
              ))}

              <label className="font-bold">Answer:</label>
              <input
                type="text"
                value={q.answer}
                onChange={(e) =>
                  handleQuizChange(qIndex, 'answer', e.target.value)
                }
                className="w-1/2 ml-2 p-1 border-2 rounded-lg"
              />
            </div>
          </div>
        ))}

        <button
          type="button"
          onClick={handleUpdate}
          className="w-fit px-4 py-2 text-white bg-green-500 rounded-lg hover:bg-green-600"
        >
          Update Quiz
        </button>
      </form>
    </div>
  );
};

export default Update;
