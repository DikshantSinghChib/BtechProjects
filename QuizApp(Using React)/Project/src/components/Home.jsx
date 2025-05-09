import React from 'react'

const Home = () => {
  return (
    <div className='bg-gradient-to-r from-teal-400 via-purple-500 to-pink-400 p-[4rem] flex justify-between items-center gap-[10rem] flex-col lg:flex-row'>
      <div className='w-[400px] h-[300px] md:w-[500px] md:h-[400px] lg:w-[700px] lg:h-[400px]'>
        <img src="/images/sectionImg.png" alt="" className='w-full h-full' />
      </div>
      <div className='flex flex-col justify-end items-start gap-[1rem] mb-16 mr-6'>
          <div className='flex gap-2'>
            <img src="/images/ideas.png" alt="" className='w-[40px] h-[35px]'/>
            <h4 className='font-bold text-2xl text-white'>Make Quiz</h4>
          </div>
          <div className='flex gap-[1rem] text-white justify-center items-center'>
            <a href="/quiz/easy" className="
                  bg-gradient-to-r from-[#56CCF2] via-[#2F80ED] to-[#56CCF2]
                  bg-[length:200%_auto] hover:bg-[position:right_center]
                  text-white text-center uppercase font-semibold
                  px-[1.5rem] py-[0.3rem] my-1 rounded-lg shadow-[0_0_20px_#eee]
                  transition-all duration-500 block
                ">Easy</a>
            <a href="/quiz/medium" className="
                  bg-gradient-to-r from-[#56CCF2] via-[#2F80ED] to-[#56CCF2]
                  bg-[length:200%_auto] hover:bg-[position:right_center]
                  text-white text-center uppercase font-semibold
                  px-[1.5rem] py-[0.3rem] my-1 rounded-lg shadow-[0_0_20px_#eee]
                  transition-all duration-500 block
                ">Medium</a>
            <a href="/quiz/hard" className="
                  bg-gradient-to-r from-[#56CCF2] via-[#2F80ED] to-[#56CCF2]
                  bg-[length:200%_auto] hover:bg-[position:right_center]
                  text-white text-center uppercase font-semibold
                  px-[1.5rem] py-[0.3rem] my-1 rounded-lg shadow-[0_0_20px_#eee]
                  transition-all duration-500 block
                ">Hard</a>
          </div>
      </div>
    </div>
  )
}

export default Home
