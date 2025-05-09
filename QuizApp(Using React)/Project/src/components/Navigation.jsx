import React from 'react'
const Navigation = () => {
    return (
        <div className='bg-gradient-to-r from-teal-400 via-purple-500 to-pink-400 py-2 flex justify-around md:justify-between items-center px-[5rem]'>
          <div className='h-[35px] w-[90px]'>
            <img src="/images/logo.png" alt="Quizzy logo" className='w-full h-full'/>
          </div>
          <div className='sm:flex gap-[1rem] text-white justify-center items-center hidden sm:block'>
            <a href="/" className='bg-gradient-to-r from-[#56CCF2] via-[#2F80ED] to-[#56CCF2]
                  bg-[length:200%_auto] hover:bg-[position:right_center]
                  text-white text-center uppercase font-semibold
                  px-[1.5rem] py-[0.3rem] my-1 rounded-lg shadow-[0_0_20px_#eee]
                  transition-all duration-500 block'>Home</a>
            <a href="/cards" className='bg-gradient-to-r from-[#56CCF2] via-[#2F80ED] to-[#56CCF2]
                  bg-[length:200%_auto] hover:bg-[position:right_center]
                  text-white text-center uppercase font-semibold
                  px-[1.5rem] py-[0.3rem] my-1 rounded-lg shadow-[0_0_20px_#eee]
                  transition-all duration-500 block'>Quizes</a>
          </div>
        </div>
      );
    
}
 
export default Navigation
