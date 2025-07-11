import { Link } from 'react-router-dom';

export default function HomePage() {
  return (
    <div className="bg-gradient-to-b from-[#121212] to-[#1E1E1E] text-white min-h-screen">
      {/* Header/Navigation */}
      {/* <header className="px-6 py-4 border-b border-gray-800">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="flex items-center">
            <span className="text-[#BB86FC] text-2xl font-bold">NeuroScanAI</span>
          </div>
          <nav className="hidden md:flex space-x-8">
            <a href="#about" className="text-gray-300 hover:text-[#BB86FC] transition">About</a>
            <a href="#features" className="text-gray-300 hover:text-[#BB86FC] transition">Features</a>
            <a href="#how-it-works" className="text-gray-300 hover:text-[#BB86FC] transition">How It Works</a>
            <Link to="/detect" className="bg-[#BB86FC] text-[#121212] px-4 py-2 rounded-lg font-medium hover:bg-opacity-90 transition">
              Start Diagnosis
            </Link>
          </nav>
        </div>
      </header> */}

      {/* Hero Section */}
      <section className="relative py-20 px-6">
        <div className="max-w-7xl mx-auto grid md:grid-cols-2 gap-12 items-center">
          <div className="md:pr-12">
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-white leading-tight">
              <span className="text-[#BB86FC]">Early Detection</span> Is Key To Alzheimer's Care
            </h1>
            <p className="mt-6 text-lg text-gray-300 max-w-lg">
              Our advanced AI system analyzes brain MRI scans to detect early signs of Alzheimer's disease with high accuracy, helping patients get timely treatment.
            </p>
            <div className="mt-8 flex flex-col sm:flex-row gap-4">
              <Link 
                to="/detect" 
                className="bg-[#BB86FC] text-[#121212] px-6 py-3 rounded-lg font-semibold hover:bg-opacity-90 transition text-center"
              >
                Start Diagnosis
              </Link>
              <a 
                href="#how-it-works" 
                className="border border-[#BB86FC] text-[#BB86FC] px-6 py-3 rounded-lg font-semibold hover:bg-[#BB86FC] hover:bg-opacity-10 transition text-center"
              >
                Learn How It Works
              </a>
            </div>
          </div>
          <div className="relative">
            <div className="bg-[#2A2A2A] p-2 rounded-xl shadow-2xl overflow-hidden border border-gray-800">
              <img 
                src="https://prod-images-static.radiopaedia.org/images/13656005/bd937738ad6223a03f8aedcf4920a7_big_gallery.jpeg" 
                alt="Brain MRI Scan" 
                className="w-full h-auto rounded-lg"
              />
              <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-tr from-[#BB86FC] to-transparent opacity-20 rounded-xl"></div>
            </div>
            <div className="absolute -bottom-4 -right-4 bg-[#1E1E1E] p-4 rounded-lg border border-gray-800 shadow-xl">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-green-400"></div>
                <p className="text-sm font-medium">97% Accuracy Rate</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-10 px-6 border-y border-gray-800">
        <div className="max-w-7xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-8">
          {stats.map((stat, index) => (
            <div key={index} className="text-center">
              <p className="text-3xl md:text-4xl font-bold text-[#BB86FC]">{stat.value}</p>
              <p className="text-sm text-gray-400 mt-1">{stat.label}</p>
            </div>
          ))}
        </div>
      </section>

      {/* About Section */}
<section id="about" className="py-20 px-6">
  <div className="max-w-7xl mx-auto">
    <div className="text-center mb-16">
      <h2 className="text-3xl md:text-4xl font-bold text-[#BB86FC]">About Our Technology</h2>
      <p className="mt-4 text-lg text-gray-300 max-w-2xl mx-auto">
        We combine advanced deep learning with medical expertise to provide accurate Alzheimer's detection and personalized care recommendations.
      </p>
    </div>

    <div className="grid md:grid-cols-2 gap-12 items-center">
      {/* Text with number badges */}
      <div className="order-2 md:order-1">
        <h3 className="text-2xl font-semibold mb-4">How Our AI Works</h3>
        <div className="space-y-6">
          {aboutPoints.map((point, index) => (
            <div key={index} className="flex gap-3">
              <div className="flex-shrink-0 w-10 h-10 rounded-full bg-[#BB86FC] text-black font-extrabold flex items-center justify-center shadow-md">
                {index + 1}
              </div>
              <div>
                <h4 className="font-medium text-white">{point.title}</h4>
                <p className="text-gray-400 mt-1">{point.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Right side images */}
      <div className="order-1 md:order-2 grid grid-cols-2 gap-4">
        <div className="bg-[#2A2A2A] p-2 rounded-lg aspect-square">
          <img 
            src="https://case.edu/med/neurology/NR/flairdwicom.jpg" 
            alt="Brain MRI comparison" 
            className="w-full h-full object-cover rounded"
          />
        </div>
        <div className="bg-[#2A2A2A] p-2 rounded-lg row-span-2">
          <img 
            src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRmp2ImzEGTQcgu7ZMMr_367caZ6qQ22pORlg&s" 
            alt="AI analyzing brain" 
            className="w-full h-full object-cover rounded"
          />
        </div>
        <div className="bg-[#2A2A2A] p-2 rounded-lg">
          <img 
            src="https://magazine.medlineplus.gov/images/uploads/main_images/brainscan-v2-sp18.jpg" 
            alt="Brain scan technology" 
            className="w-full h-full object-cover rounded"
          />
        </div>
      </div>
    </div>
  </div>
</section>


      {/* Features Section */}
<section id="features" className="py-20 px-6 bg-[#161616]">
  <div className="max-w-7xl mx-auto">
    <div className="text-center mb-16">
      <h2 className="text-3xl md:text-4xl font-bold text-[#BB86FC]">Our Features</h2>
      <p className="mt-4 text-lg text-gray-300 max-w-2xl mx-auto">
        Comprehensive tools for Alzheimer's detection, monitoring, and care recommendations
      </p>
    </div>
    
    <div className="grid md:grid-cols-3 gap-8">
      {features.map((feature, index) => (
        <div
          key={index}
          className="bg-[#1E1E1E] rounded-xl p-6 shadow-lg border border-gray-800 hover:border-[#BB86FC] transition-all duration-300 hover:translate-y-[-5px]"
        >
          <div className="w-16 h-16 rounded-full bg-[#BB86FC] flex items-center justify-center mb-6 shadow-md">
            {/* Updated: Stronger visibility and clearer size */}
            <feature.icon className="w-7 h-7 text-black" />
          </div>
          <h3 className="text-xl font-semibold mb-3">{feature.title}</h3>
          <p className="text-gray-400">{feature.description}</p>
        </div>
      ))}
    </div>
  </div>
</section>


      {/* How It Works Section */}
      <section id="how-it-works" className="py-20 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-[#BB86FC]">How It Works</h2>
            <p className="mt-4 text-lg text-gray-300 max-w-2xl mx-auto">
              Our simple 3-step process makes Alzheimer's detection accessible and straightforward
            </p>
          </div>
          
          <div className="relative">
            <div className="absolute top-24 left-[50%] w-0.5 h-[calc(100%-6rem)] bg-gray-800 hidden md:block"></div>
            
            <div className="space-y-24">
              {steps.map((step, index) => (
                <div key={index} className="relative grid md:grid-cols-2 gap-8 items-center">
                  <div className={`md:pr-16 ${index % 2 === 1 ? 'md:order-2' : ''}`}>
                    <div className="hidden md:flex absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-12 h-12 rounded-full bg-[#BB86FC] text-[#121212] font-bold items-center justify-center shadow-lg z-10">
                      {index + 1}
                    </div>

                    <h3 className="text-2xl font-semibold mb-4">{step.title}</h3>
                    <p className="text-gray-400">{step.description}</p>
                  </div>
                  <div className={`bg-[#2A2A2A] p-3 rounded-xl shadow-lg ${index % 2 === 1 ? 'md:order-1' : ''}`}>
                    <img 
                      src={step.image} 
                      alt={step.title} 
                      className="w-full h-auto rounded-lg"
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          <div className="mt-16 text-center">
            <Link 
              to="/detect" 
              className="bg-[#BB86FC] text-[#121212] px-8 py-4 rounded-lg font-semibold hover:bg-opacity-90 transition inline-block"
            >
              Start Your Diagnosis Now
            </Link>
          </div>
        </div>
      </section>

      {/* Team Section */}
<section className="py-20 px-6 bg-[#161616]">
  <div className="max-w-7xl mx-auto">
    <div className="text-center mb-16">
      <h2 className="text-3xl md:text-4xl font-bold text-[#BB86FC]">Meet Our Team</h2>
      <p className="mt-4 text-lg text-gray-300 max-w-2xl mx-auto">
        Experts combining medical knowledge with cutting-edge AI technology
      </p>
    </div>

    <div className="grid md:grid-cols-2 gap-8">
      {team.map((member, index) => (
        <div key={index} className="bg-[#1E1E1E] rounded-xl overflow-hidden shadow-lg hover:shadow-xl transition-all duration-300 hover:translate-y-[-5px]">
          <div className="p-6">
            <h3 className="text-xl font-semibold">{member.name}</h3>
            <p className="text-[#BB86FC] mb-3">{member.role}</p>
            <p className="text-gray-400">
              {member.bio || "Research expert combining medical knowledge with AI technology to advance Alzheimer's detection."}
            </p>
          </div>
        </div>
      ))}
    </div>
  </div>
</section>

      {/* CTA Section */}
      <section className="py-16 px-6">
        <div className="max-w-5xl mx-auto bg-gradient-to-r from-[#BB86FC] to-[#9573E8] rounded-2xl p-8 md:p-12 text-center">
          <h2 className="text-3xl md:text-4xl font-bold text-[#121212] mb-6">Ready to detect Alzheimer's early?</h2>
          <p className="text-gray-800 text-lg mb-8 max-w-2xl mx-auto">
            Start using our AI-powered tool to analyze brain MRI scans and get instant results with hospital recommendations.
          </p>
          <Link 
            to="/detect" 
            className="bg-[#121212] text-white px-8 py-4 rounded-lg font-semibold hover:bg-opacity-90 transition inline-block"
          >
            Start Free Analysis
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-[#121212] py-12 px-6 border-t border-gray-800">
        <div className="max-w-7xl mx-auto">
          <div className="md:flex md:justify-between">
            <div className="mb-8 md:mb-0">
              <h3 className="text-[#BB86FC] text-2xl font-bold mb-4">NeuroScanAI</h3>
              <p className="text-gray-400 max-w-xs">Advanced AI-powered Alzheimer's detection and hospital recommendation system.</p>
            </div>
            
            <div className="grid grid-cols-2 md:grid-cols-3 gap-8">
              <div>
                <h4 className="text-lg font-semibold mb-4">Quick Links</h4>
                <ul className="space-y-2">
                  <li><a href="#about" className="text-gray-400 hover:text-[#BB86FC] transition">About</a></li>
                  <li><a href="#features" className="text-gray-400 hover:text-[#BB86FC] transition">Features</a></li>
                  <li><a href="#how-it-works" className="text-gray-400 hover:text-[#BB86FC] transition">How It Works</a></li>
                </ul>
              </div>
              
              <div>
                <h4 className="text-lg font-semibold mb-4">Resources</h4>
                <ul className="space-y-2">
                  <li><a href="#" className="text-gray-400 hover:text-[#BB86FC] transition">Blog</a></li>
                  <li><a href="#" className="text-gray-400 hover:text-[#BB86FC] transition">Research</a></li>
                  <li><a href="#" className="text-gray-400 hover:text-[#BB86FC] transition">Help Center</a></li>
                </ul>
              </div>
              
              <div>
                <h4 className="text-lg font-semibold mb-4"></h4>
                <ul className="space-y-2">
                  <li className="text-gray-400"></li>
                  <li className="text-gray-400"></li>
                </ul>
              </div>
            </div>
          </div>
          
          <div className="mt-12 pt-8 border-t border-gray-800 text-center md:flex md:justify-between md:text-left">
            <p className="text-gray-500">&copy; 2025 NeuroScan AI. All rights reserved.</p>
            <p className="text-gray-500 mt-2 md:mt-0">Created by Eshan Vijay & Siddhant Patil</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

// Data for the homepage
const stats = [
  { value: "97%", label: "Accuracy Rate" },
  { value: "24/7", label: "Availability" },
  { value: "3-5s", label: "Processing Time" },
  { value: "150+", label: "Hospital Partners" }
];

const aboutPoints = [
  { 
    title: "Advanced Image Processing", 
    description: "Our AI uses specialized image processing algorithms designed for medical scans."
  },
  { 
    title: "Deep Learning Classification", 
    description: "Multiple neural networks work together to identify patterns in brain structure."
  },
  { 
    title: "Medical Expert Verification", 
    description: "Our system is trained and verified by neurologists with decades of experience."
  }
];

const features = [
  { 
    title: "AI-Based Diagnosis", 
    description: "Advanced detection system identifies early signs of Alzheimer's with high accuracy.", 
    icon: () => (
      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
    )
  },
  { 
    title: "Hospital Finder", 
    description: "Instantly find specialized hospitals and treatment centers near you for Alzheimer's care.", 
    icon: () => (
      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
      </svg>
    )
  },
  { 
    title: "Detailed Reports", 
    description: "Receive comprehensive analysis of brain scans with personalized treatment suggestions.", 
    icon: () => (
      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    )
  }
];

const steps = [
  {
    title: "Upload or Capture an MRI Scan",
    description: "Either upload an existing brain MRI scan or use your device's camera to capture an image of the scan. Our system works with most common scan formats.",
    image: "/assets/image1.png"
  },
  {
    title: "AI Analysis of Brain Structure",
    description: "Our advanced AI algorithm analyzes the brain structure in the MRI, looking for specific patterns and changes that are indicative of different stages of Alzheimer's.",
    image: '/assets/image2.png'
  },
  {
    title: "Get Instant Results and Recommendations",
    description: "Within seconds, receive a detailed assessment showing the detected stage of Alzheimer's along with symptoms, treatment approaches, and recommended specialized hospitals.",
    image: "/assets/image3.png"
  }
];

const team = [
  { 
    name: "Eshan Vijay", 
    role: "AI & Machine Learning Lead", 
    bio: "Expert in developing and training neural networks for medical image analysis. Created the core model behind our Alzheimer's detection system."
  },
  { 
    name: "Siddhant Patil", 
    role: "UI/UX Designer & Frontend Developer", 
    bio: "Skilled designer focused on creating intuitive and accessible user interfaces. Responsible for the platform's user experience and visual design."
  }
];
