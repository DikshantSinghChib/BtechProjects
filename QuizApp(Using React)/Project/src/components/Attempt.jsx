import React, { useEffect, useState } from 'react';
import { useSelector } from 'react-redux';
import { useParams, useNavigate } from "react-router-dom";
import { PieChart } from 'react-minimal-pie-chart';

const Attempt = () => {
  const quizes = useSelector((state) => state.quiz.quizes);
  const { id } = useParams();
  const navigate = useNavigate();

  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [quiz, setQuiz] = useState([]);
  const [answers, setAnswers] = useState({});
  const [score, setScore] = useState(null);
  const [submitted, setSubmitted] = useState(false);

  useEffect(() => {
    const foundQuiz = quizes.find((q) => q.id === id);
    if (foundQuiz) {
      setTitle(foundQuiz.title);
      setDescription(foundQuiz.description || '');
      setQuiz(foundQuiz.quiz);
    }
  }, [id, quizes]);

  const handleOptionSelect = (qIndex, optionValue) => {
    setAnswers((prev) => ({ ...prev, [qIndex]: optionValue }));
  };

  const handleSubmit = () => {
    let totalScore = 0;
    quiz.forEach((q, index) => {
      if (answers[index] === q.answer) {
        totalScore += 1;
      }
    });
    setScore(totalScore);
    setSubmitted(true);
  };

  const handleReturn = () => {
    navigate('/');
  };

  return (
    <div className='flex flex-col max-w-[1000px] mx-[2rem] lg:mx-auto mt-4 border-4 rounded-md p-5 mb-5'>
      <p className="text-[2rem] mb-2 font-bold text-center">{title}</p>
      <div>
        {quiz.map((q, qIndex) => (
          <div key={qIndex} className="flex flex-col gap-2 mb-6">
            <h2 className="font-bold text-lg">Q {qIndex + 1}: {q.question} ?</h2>
            <div className="ml-[4rem] mt-2">
              {q.options.map((opt, optIndex) => (
                <div key={optIndex} className="mb-2">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="radio"
                      name={`question-${qIndex}`}
                      value={opt}
                      checked={answers[qIndex] === opt}
                      onChange={() => handleOptionSelect(qIndex, opt)}
                    />
                    {opt}
                  </label>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {!submitted && (
        <button
          onClick={handleSubmit}
          className="bg-blue-600 text-white px-4 py-2 rounded-md self-start"
        >
          Submit
        </button>
      )}

      {submitted && score !== null && (
        <>
          <div className="flex flex-col items-center mt-2 mb-1">
            <div className="relative w-[280px] h-[180px]">
              <PieChart
                data={[
                  { title: 'Correct', value: score, color: '#22c55e' },   // Tailwind green-500
                  { title: 'Incorrect', value: quiz.length - score, color: '#ef4444' }, // Tailwind red-500
                ]}
                totalValue={quiz.length}
                startAngle={180}
                lengthAngle={180}
                lineWidth={20}
                animate
                style={{ height: '200px' }}
              />
            
              {/* Fake Center Label */}
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-xl font-bold">
                  {score} / {quiz.length}
                </div>
              </div>
            </div>
            
            <div className="mt-[-50px] flex gap-6">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-green-500 rounded-full"></div>
                <span>Correct</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-red-500 rounded-full"></div>
                <span>Incorrect</span>
              </div>
            </div>
          </div>
          <button
            onClick={handleReturn}
            className="bg-blue-600 text-white px-4 py-2 mt-4 rounded-md self-center"
          >
            Return
          </button>
        </>
      )}
    </div>
  );
};

export default Attempt;
