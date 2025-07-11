export default function BlogPage() {
  const blogs = [
  {
    title: "Understanding Alzheimer's: Causes and Symptoms",
    author: "Dr. A. Sharma",
    date: "Feb 12, 2025",
    description:
      "Alzheimer’s disease is a progressive neurological disorder that affects memory and thinking abilities. Learn about the risk factors, early symptoms, and possible treatment options available today.",
    image: "/assets/blog1.png",  // ✅ Relative path from public folder
  },
  {
    title: "Early Signs of Alzheimer’s and How to Detect Them",
    author: "Dr. P. Verma",
    date: "Feb 10, 2025",
    description:
      "Early detection of Alzheimer’s can help in managing the disease better. This blog covers common symptoms and early-stage diagnosis techniques that can be helpful.",
    image: "/assets/blog2.png",
  },
];


  return (
    <section className="py-20 px-6 bg-[#121212] text-white">
      <div className="max-w-6xl mx-auto">
        <h2 className="text-4xl font-bold text-[#BB86FC] mb-12 text-center">Our Blog</h2>

        <div className="space-y-12">
          {blogs.map((blog, index) => (
            <div
              key={index}
              className="md:flex items-center gap-8 bg-[#1E1E1E] rounded-xl p-6 shadow-lg hover:shadow-xl transition-all duration-300"
            >
              <div className="md:w-1/2 mb-4 md:mb-0">
                <img
                  src={blog.image}
                  alt={blog.title}
                  className="w-full h-auto rounded-lg object-cover"
                />
              </div>
              <div className="md:w-1/2">
                <h3 className="text-2xl font-semibold text-[#BB86FC] mb-2">{blog.title}</h3>
                <p className="text-sm text-gray-400 mb-2">
                  By <span className="text-white font-medium">{blog.author}</span> — {blog.date}
                </p>
                <p className="text-gray-300">{blog.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
