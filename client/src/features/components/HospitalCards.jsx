export default function HospitalPage() {
    const hospitals = [
      {
        name: "Medicover Hospitals",
        location: "Mumbai, India",
        contact: "+91 9876543210",
        image: "/assets/hospital2.png", // Replace with actual hospital image
        services: "Specialized memory care, Neurology department, Early diagnosis support.",
      },
      {
        name: "Kokilaben Dhirubhai Ambani Hospital and Medical Research Institute",
        location: "Mumbai, India",
        contact: "+91 9876543220",
        image: "/assets/hospital1.png",
        services: "Cognitive therapy, Alzheimer’s treatment, 24/7 medical assistance.",
      },
    ];
  
    return (
      <div className="min-h-screen bg-[#121212] text-[#E0E0E0] p-6">
        <h1 className="text-3xl font-bold text-[#BB86FC] mb-6">Hospitals Treating Alzheimer’s</h1>
        <div className="grid md:grid-cols-2 gap-6">
          {hospitals.map((hospital, index) => (
            <div key={index} className="bg-[#1E1E1E] p-6 rounded-2xl shadow-lg">
              <div className="w-full h-60 bg-black flex items-center justify-center rounded-lg mb-4 overflow-hidden">
  <img src={hospital.image} alt={hospital.name} className="h-full object-contain" />
</div>


              <h2 className="text-xl font-semibold text-[#BB86FC]">{hospital.name}</h2>
              <p className="text-[#BDBDBD]">📍 {hospital.location}</p>
              <p className="text-[#BDBDBD]">📞 {hospital.contact}</p>
              <p className="mt-2 text-[#E0E0E0]">{hospital.services}</p>
              <button className="mt-4 bg-[#BB86FC] text-[#121212] px-4 py-2 rounded-lg">View Details</button>
            </div>
          ))}
        </div>
      </div>
    );
  }
  