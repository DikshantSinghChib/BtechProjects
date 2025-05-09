import './App.css'
import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import Navigation from './components/Navigation'
import Home from './components/Home';
import Attempt from './components/Attempt';
import MakeEasy from './components/MakeEasy';
import MakeMedium from './components/MakeMedium';
import MakeHard from './components/MakeHard';
import QuizCard from './components/QuizCard';
import About from './components/About';
import Update from './components/Update';
import Footer from './components/footer';
const router = createBrowserRouter([
  {
    path: '/',
    element: (
      <div>
        <Navigation />
        <Home />
        <QuizCard/>
        <Footer/>
      </div>
    ),
  },
  {
    path: '/quiz/easy',
    element: (
      <div>
        <Navigation />
        <MakeEasy />
        <Footer/>
      </div>
    ),
  },
  {
    path: '/quiz/medium',
    element: (
      <div>
        <Navigation />
        <MakeMedium />
        <Footer/>
      </div>
    ),
  },
  {
    path: '/quiz/hard',
    element: (
      <div>
        <Navigation />
        <MakeHard />
        <Footer/>
      </div>
    ),
  },
  {
    path: '/cards',
    element: (
      <div>
        <Navigation />
        <QuizCard />
      </div>
    ),
  },
  {
    path: '/about',
    element: (
      <div>
        <Navigation />
        <About />
        <Footer/>
      </div>
    ),
  },
  {
    path: '/quiz/:id',
    element: (
      <div>
        <Navigation/>
        <Update />
        <Footer/>
      </div>
    ),
  },
  {
    path: '/quiz/attempt/:id',
    element: (
      <div>
        <Navigation/>
        <Attempt />
        <Footer/>
      </div>
    ),
  },
]);
function App() {

  return (
    <div>
      <RouterProvider router={router} />
    </div>
  )
}

export default App
