"""
Dataset Generation and Validation Script
Generates exactly 300 unique, balanced, diverse benchmark queries:
- 100 Easy (Level 1)
- 100 Medium (Level 2)
- 100 Hard (Level 3)
"""

import json
import os
import pandas as pd

def build_unique_dataset():
    # 100 Distinct Easy Queries
    easy = [
        # CS & Tech (25)
        ("What does HTML stand for?", "Computer Science", "HyperText Markup Language", "Direct factual lookup"),
        ("What is the primary function of RAM in a computer?", "Computer Science", "Temporary working memory for fast CPU access.", "Basic definition"),
        ("What does CPU stand for?", "Computer Science", "Central Processing Unit", "Direct factual lookup"),
        ("Define what an operating system is in computing.", "Computer Science", "Software managing hardware and software resources.", "Basic definition"),
        ("What symbol is used to start a single-line comment in Python?", "Computer Science", "The hash (#) symbol.", "Syntax lookup"),
        ("What does SQL stand for?", "Computer Science", "Structured Query Language", "Direct factual lookup"),
        ("What file extension is used for standard Python code files?", "Computer Science", ".py", "Direct factual lookup"),
        ("What does GUI stand for in software systems?", "Computer Science", "Graphical User Interface", "Direct factual lookup"),
        ("What is a boolean data type in computer programming?", "Computer Science", "A binary data type holding True or False.", "Basic definition"),
        ("Which keyword is used to declare a function in Python?", "Computer Science", "The 'def' keyword.", "Syntax lookup"),
        ("What is an IP address?", "Computer Science", "A unique numerical network identifier for a device.", "Basic definition"),
        ("What does URL stand for in web technology?", "Computer Science", "Uniform Resource Locator", "Direct factual lookup"),
        ("What is the purpose of the print() function in Python?", "Computer Science", "Outputs data to the standard console.", "Basic explanation"),
        ("What is Git used for in software development?", "Computer Science", "Distributed source code version control.", "Basic definition"),
        ("What does CSS stand for in web development?", "Computer Science", "Cascading Style Sheets", "Direct factual lookup"),
        ("What is an array in computer programming?", "Computer Science", "A collection of items stored at contiguous memory locations.", "Basic definition"),
        ("What does HTTP stand for in networking?", "Computer Science", "HyperText Transfer Protocol", "Direct factual lookup"),
        ("What is an integer in computer science?", "Computer Science", "A whole number without fractional components.", "Basic definition"),
        ("What is a syntax error in coding?", "Computer Science", "An error caused by violating language grammar rules.", "Basic definition"),
        ("What is an algorithm in computing?", "Computer Science", "A step-by-step set of instructions for solving a problem.", "Basic definition"),
        ("What does API stand for in software engineering?", "Computer Science", "Application Programming Interface", "Direct factual lookup"),
        ("What is the binary representation of decimal 5?", "Computer Science", "101", "Basic conversion"),
        ("What is a loop in computer programming?", "Computer Science", "A control structure that repeats a block of code.", "Basic definition"),
        ("What does IDE stand for in software development?", "Computer Science", "Integrated Development Environment", "Direct factual lookup"),
        ("What is a string in programming languages?", "Computer Science", "A sequence of characters used to represent text.", "Basic definition"),

        # Mathematics & Numbers (25)
        ("What is 15 multiplied by 8?", "Mathematics", "120", "Simple multiplication"),
        ("What is the square root of 144?", "Mathematics", "12", "Basic arithmetic"),
        ("Calculate 25 percent of 200.", "Mathematics", "50", "Basic percentage calculation"),
        ("What is 7 cubed (7^3)?", "Mathematics", "343", "Basic power calculation"),
        ("Solve the linear equation: 2x + 6 = 14.", "Mathematics", "x = 4", "Single-variable linear equation"),
        ("What is the perimeter of a rectangle with length 8 and width 5?", "Mathematics", "26", "Geometric formula"),
        ("What is the value of Pi rounded to two decimal places?", "Mathematics", "3.14", "Mathematical constant"),
        ("Convert 5 kilometers to meters.", "Mathematics", "5,000 meters", "Metric unit conversion"),
        ("How many degrees are in a right angle?", "Mathematics", "90 degrees", "Geometric definition"),
        ("What is 100 divided by 4?", "Mathematics", "25", "Basic division"),
        ("What is the sum of interior angles in any triangle?", "Mathematics", "180 degrees", "Geometric fact"),
        ("Convert 2 hours into minutes.", "Mathematics", "120 minutes", "Time conversion"),
        ("What is the greatest common divisor of 12 and 18?", "Mathematics", "6", "Arithmetic factor lookup"),
        ("What is the median of the numbers 3, 7, 9, 12, 15?", "Mathematics", "9", "Statistical median calculation"),
        ("What is 2 raised to the power of 8?", "Mathematics", "256", "Power calculation"),
        ("Calculate the area of a square with a side length of 9 centimeters.", "Mathematics", "81 sq cm", "Geometric formula"),
        ("What is 50 percent of 80?", "Mathematics", "40", "Percentage arithmetic"),
        ("What is the absolute value of -42?", "Mathematics", "42", "Arithmetic property"),
        ("How many sides does an octagon have?", "Mathematics", "8 sides", "Geometric fact"),
        ("What is 13 minus 27?", "Mathematics", "-14", "Negative arithmetic"),
        ("What is the factorial of 4 (4!)?", "Mathematics", "24", "Factorial calculation"),
        ("How many minutes are in 3.5 hours?", "Mathematics", "210 minutes", "Unit conversion"),
        ("What is 11 squared?", "Mathematics", "121", "Square calculation"),
        ("What is the reciprocal of 4?", "Mathematics", "0.25 (1/4)", "Reciprocal definition"),
        ("What is the volume of a cube with edge length 3 cm?", "Mathematics", "27 cubic cm", "Geometric formula"),

        # Natural Sciences (25)
        ("What is the chemical formula for pure water?", "Science", "H2O", "Direct chemical formula"),
        ("Which planet is commonly called the Red Planet?", "Science", "Mars", "Astronomical fact"),
        ("What gas do green plants absorb from the atmosphere for photosynthesis?", "Science", "Carbon dioxide (CO2)", "Biological fact"),
        ("What is the freezing point of water in degrees Celsius?", "Science", "0 degrees C", "Physical constant"),
        ("Which muscular organ pumps blood throughout the human body?", "Science", "The heart", "Biological fact"),
        ("What is the approximate speed of light in a vacuum in km/s?", "Science", "300,000 km/s", "Physical constant"),
        ("Which chemical element has the atomic number 1?", "Science", "Hydrogen", "Periodic table fact"),
        ("What is the largest organ of the human body?", "Science", "Skin", "Anatomical fact"),
        ("What fundamental force pulls physical objects toward Earth's center?", "Science", "Gravity", "Physical concept"),
        ("What is the dense center of an atom called?", "Science", "The nucleus", "Physical definition"),
        ("What green pigment is essential for plant photosynthesis?", "Science", "Chlorophyll", "Biological fact"),
        ("How many recognized planets are in our Solar System?", "Science", "8 planets", "Astronomical fact"),
        ("What state of matter has a fixed volume but conforms to container shape?", "Science", "Liquid", "Physical state definition"),
        ("What is the boiling point of pure water at 1 atm in Celsius?", "Science", "100 degrees C", "Physical constant"),
        ("What gas accounts for roughly 78 percent of Earth's atmosphere?", "Science", "Nitrogen", "Atmospheric science fact"),
        ("Which blood cells are primarily responsible for fighting infections?", "Science", "White blood cells", "Biological fact"),
        ("What is the chemical symbol for gold?", "Science", "Au", "Chemical symbol lookup"),
        ("What is the chemical symbol for table salt?", "Science", "NaCl", "Chemical formula"),
        ("What is the name of our home galaxy?", "Science", "The Milky Way", "Astronomical fact"),
        ("What part of the human skeleton protects the brain?", "Science", "The skull (cranium)", "Anatomical fact"),
        ("What process converts liquid water into vapor?", "Science", "Evaporation", "Physical process"),
        ("What is the hardest naturally occurring mineral?", "Science", "Diamond", "Geological fact"),
        ("Which subatomic particle carries a negative electric charge?", "Science", "The electron", "Physical particle fact"),
        ("What layer of Earth lies directly beneath the crust?", "Science", "The mantle", "Geological fact"),
        ("What is the chemical symbol for iron?", "Science", "Fe", "Chemical symbol lookup"),

        # Humanities, Geography & General Knowledge (25)
        ("Who painted the famous masterpiece Mona Lisa?", "Humanities", "Leonardo da Vinci", "Historical art fact"),
        ("What is the capital city of France?", "Geography", "Paris", "Geographic capital lookup"),
        ("In what year did the ocean liner Titanic sink?", "History", "1912", "Historical fact"),
        ("What is the capital city of Japan?", "Geography", "Tokyo", "Geographic capital lookup"),
        ("Who authored the classic tragedy Romeo and Juliet?", "Humanities", "William Shakespeare", "Literary lookup"),
        ("On which continent is the Sahara Desert situated?", "Geography", "Africa", "Geographic fact"),
        ("What is the official currency of the United Kingdom?", "General", "Pound Sterling (GBP)", "Financial currency lookup"),
        ("Who served as the first President of the United States?", "History", "George Washington", "Historical fact"),
        ("Which body of water is the largest ocean on Earth?", "Geography", "The Pacific Ocean", "Geographic fact"),
        ("What language is predominantly spoken across Brazil?", "Geography", "Portuguese", "Linguistic fact"),
        ("Who is credited with the discovery of penicillin in 1928?", "History", "Alexander Fleming", "Scientific history"),
        ("What is the capital city of Australia?", "Geography", "Canberra", "Geographic capital lookup"),
        ("In which country are the Great Pyramids of Giza located?", "Geography", "Egypt", "Geographic fact"),
        ("Who composed the ancient Greek epics Iliad and Odyssey?", "Humanities", "Homer", "Literary fact"),
        ("In what year did the Second World War officially conclude?", "History", "1945", "Historical fact"),
        ("What is the highest mountain peak above sea level on Earth?", "Geography", "Mount Everest", "Geographic fact"),
        ("What is the capital city of Germany?", "Geography", "Berlin", "Geographic capital lookup"),
        ("What is the capital city of Canada?", "Geography", "Ottawa", "Geographic capital lookup"),
        ("What is the capital city of Italy?", "Geography", "Rome", "Geographic capital lookup"),
        ("What is the capital city of Spain?", "Geography", "Madrid", "Geographic capital lookup"),
        ("What is the capital city of Egypt?", "Geography", "Cairo", "Geographic capital lookup"),
        ("What is the capital city of China?", "Geography", "Beijing", "Geographic capital lookup"),
        ("What is the official national animal of India?", "General", "The Bengal Tiger", "General knowledge fact"),
        ("What instrument is used by meteorologists to measure air pressure?", "General", "A barometer", "Instrument lookup"),
        ("How many days are in a standard leap year?", "General", "366 days", "Calendar fact")
    ]

    # 100 Distinct Medium Queries
    medium = [
        # CS & Data Science (30)
        ("Explain the fundamental difference between supervised and unsupervised machine learning.", "Computer Science", "Supervised learning relies on labeled targets, whereas unsupervised learning discovers patterns in unlabeled data.", "Conceptual comparison"),
        ("What is the difference between a process and a thread in operating systems?", "Computer Science", "Processes have independent memory address spaces; threads within a process share memory and heap.", "OS comparison"),
        ("Explain the mechanism of binary search and compare its time complexity with linear search.", "Computer Science", "Binary search halves sorted search intervals achieving O(log n); linear search checks elements sequentially taking O(n).", "Algorithm analysis"),
        ("Compare SQL relational databases and NoSQL document stores regarding schema flexibility and scaling.", "Computer Science", "SQL enforces strict relational schemas and ACID; NoSQL provides flexible JSON schemas and horizontal scaling.", "Database architecture"),
        ("Explain the core architectural principles of REST APIs and list four standard HTTP verbs.", "Computer Science", "REST is stateless and resource-oriented. Common verbs: GET, POST, PUT, DELETE.", "Web architecture"),
        ("How do Docker containers differ from hypervisor-based virtual machines?", "Computer Science", "Containers share the host kernel and isolate processes; VMs virtualize full hardware and run guest OS instances.", "DevOps comparison"),
        ("Explain the distinction between mutable and immutable data types in Python with examples.", "Computer Science", "Mutable types (lists, dicts) can be modified in-place; immutable types (strings, tuples) cannot be altered once created.", "Programming concepts"),
        ("Explain how hash collisions are resolved using separate chaining versus open addressing.", "Computer Science", "Chaining stores colliding records in bucket linked lists; open addressing probes adjacent array slots.", "Data structures"),
        ("What is overfitting in predictive models and what are two common regularization techniques to mitigate it?", "Machine Learning", "Overfitting is memorizing noise. Mitigation includes L1/L2 weight penalties and dropout.", "Model optimization"),
        ("Explain the difference between synchronous and asynchronous I/O execution in web services.", "Computer Science", "Synchronous I/O blocks calling threads; asynchronous I/O utilizes non-blocking event loops and futures.", "Concurrency concepts"),
        ("Compare monolithic architectures with microservice architectures for engineering teams.", "Software Engineering", "Monoliths simplify early deployments and debugging; microservices offer independent scaling but add network complexity.", "Architectural trade-offs"),
        ("Explain the trade-offs between TCP and UDP transport layer protocols with appropriate use cases.", "Networking", "TCP provides ordered, reliable streams (web/HTTP); UDP minimizes latency without packet acknowledgments (gaming/DNS).", "Protocol comparison"),
        ("How does a Convolutional Neural Network preserve spatial features compared to dense layers?", "Machine Learning", "CNNs utilize local receptive kernels with weight sharing, capturing 2D spatial hierarchies efficiently.", "Deep learning concepts"),
        ("What is the principle of least privilege in access management and how is it applied in cloud IAM?", "Cybersecurity", "Entities receive only the bare permissions needed to complete tasks, implemented via granular role-based policies.", "Security principles"),
        ("Explain the trade-off between Precision and Recall in classification and when to favor each.", "Machine Learning", "Precision measures accuracy of positive flags; recall measures retrieval coverage. Favor recall when false negatives are dangerous.", "Metric analysis"),
        ("Distinguish between concurrency and parallelism in computer system execution.", "Computer Science", "Concurrency is structuring tasks to handle multiple things simultaneously; parallelism is physical concurrent hardware execution.", "System fundamentals"),
        ("Explain how public-key asymmetric encryption facilitates secure communication across public channels.", "Cybersecurity", "Uses mathematical keypairs where public keys encrypt and secret private keys decrypt without transmitting shared secrets.", "Cryptography"),
        ("What is database normalization and what conditions satisfy Third Normal Form (3NF)?", "Databases", "Normalization reduces data redundancy. 3NF requires 2NF compliance and no transitive non-prime dependencies.", "Database design"),
        ("Compare Git merge and Git rebase workflows, explaining advantages of each.", "DevOps", "Merge retains chronological branch history with merge commits; rebase creates a linear history by replaying commits.", "Version control"),
        ("Explain the difference between bagging and boosting ensemble learning techniques.", "Machine Learning", "Bagging averages independent models trained on bootstrapped subsets; boosting trains sequential models correcting past errors.", "Ensemble algorithms"),
        ("Describe the function of indexes in relational databases and the trade-offs of B-Tree indexing.", "Databases", "Indexes speed up read queries from O(n) to O(log n) but add write overhead and additional storage footprint.", "Storage performance"),
        ("Explain how a Transformer self-attention block computes token representations.", "Machine Learning", "It maps tokens into Query, Key, Value vectors, taking scaled dot-product attention scores to output weighted context vectors.", "Neural architectures"),
        ("What are memory leaks in garbage-collected languages and how can they occur in Python?", "Computer Science", "Leaks happen when unneeded objects remain reachable through circular references or static caches.", "Runtime diagnostics"),
        ("Explain the purpose of Docker multi-stage builds in containerized CI/CD pipelines.", "DevOps", "Separates compilation environments from minimal runtime images, shrinking image size and eliminating build tooling attack surfaces.", "Container optimization"),
        ("How does DNS resolution translate a human-readable domain into an IP address?", "Networking", "The client queries recursive resolvers, root servers, TLD servers, and authoritative nameservers to fetch DNS records.", "Network protocol"),
        ("What is the difference between deep copying and shallow copying compound objects in Python?", "Computer Science", "Shallow copying copies the container but references nested objects; deep copying recursively duplicates all nested items.", "Language mechanics"),
        ("Explain the CAP theorem and describe the trade-offs a distributed database must make.", "Distributed Systems", "In the presence of network partitions (P), a distributed system can guarantee either Consistency (C) or Availability (A).", "System theory"),
        ("Describe how message queues like Kafka or RabbitMQ decouple asynchronous distributed services.", "System Design", "Producers publish messages to topics/queues without blocking; consumers consume messages asynchronously at their own rate.", "Message passing"),
        ("What is CORS in web browsers and why does it protect web users from cross-site vulnerabilities?", "Web Security", "Cross-Origin Resource Sharing restricts frontend scripts from querying different origins unless explicitly allowed via headers.", "Browser security"),
        ("Explain the role of gradient descent optimization in training neural network parameters.", "Machine Learning", "It iteratively updates weight matrices in the opposite direction of the loss function gradient to minimize prediction error.", "Optimization theory"),

        # Mathematics & Multi-Step Calculations (25)
        ("A jacket priced at $150 receives a 20% store discount, followed by a 10% member coupon. What is the final price?", "Mathematics", "$108 ($150 * 0.80 = $120; $120 * 0.90 = $108).", "Multi-step percentage arithmetic"),
        ("A car travels at 60 mph for 2.5 hours and then 45 mph for 1.5 hours. What is the average speed for the trip?", "Mathematics", "Total distance = 217.5 miles; total time = 4.0 hours; average speed = 54.375 mph.", "Weighted average calculation"),
        ("If an investment portfolio yields a 6% annual return, how many years will it take to double using the Rule of 72?", "Finance", "72 / 6 = 12 years approximately.", "Financial rule application"),
        ("A pool measuring 25m long, 10m wide, and 2m deep is filled with water. How many liters does it contain?", "Mathematics", "Volume = 500 cubic meters = 500,000 liters.", "Dimensional geometry"),
        ("Pipe A fills a reservoir in 6 hours and Pipe B empties it in 9 hours. How long to fill it with both open?", "Mathematics", "Net rate = 1/6 - 1/9 = 1/18 reservoir/hour. It takes 18 hours.", "Work rate problem"),
        ("A firm's revenue was $400,000 in Year 1 and grew to $640,000 in Year 3. What is the compound annual growth rate?", "Finance", "CAGR = (640,000/400,000)^0.5 - 1 = (1.6)^0.5 - 1 = approx 26.49%.", "CAGR calculation"),
        ("Calculate the sample standard deviation for the dataset: [4, 8, 6, 5, 3, 7].", "Mathematics", "Mean = 5.5. Sum of squared differences = 17.5. Sample variance = 3.5. Standard deviation = approx 1.87.", "Statistical calculation"),
        ("A train 200 meters long crosses a bridge 400 meters long at 72 km/h. How many seconds does it take?", "Physics", "Speed = 20 m/s. Total distance = 600m. Time = 600 / 20 = 30 seconds.", "Kinematics word problem"),
        ("If two fair six-sided dice are rolled, what is the exact probability that their sum equals 8?", "Mathematics", "5 favorable outcomes out of 36 possibilities = 5/36 (approx 13.89%).", "Combinatorial probability"),
        ("Solve the system of linear equations: 3x + 2y = 16 and x - y = 2.", "Mathematics", "x = 4, y = 2.", "Linear system elimination"),
        ("Determine the derivative of the function f(x) = x^3 * ln(x) using the product rule.", "Mathematics", "f'(x) = 3x^2 * ln(x) + x^2.", "Calculus product rule"),
        ("A cylindrical tank has radius 3 meters and height 7 meters. What is its total surface area including caps?", "Mathematics", "Total Area = 2*pi*r*h + 2*pi*r^2 = 42*pi + 18*pi = 60*pi sq meters (approx 188.5 sq m).", "Geometric surface area"),
        ("Evaluate the definite integral of (3x^2 - 4x + 1) dx evaluated from x=0 to x=3.", "Mathematics", "[x^3 - 2x^2 + x] from 0 to 3 = 27 - 18 + 3 = 12.", "Definite integral calculation"),
        ("If a principal of $1,000 compounds quarterly at an annual nominal rate of 8%, what is the balance after 2 years?", "Finance", "A = 1000 * (1 + 0.08/4)^8 = 1000 * (1.02)^8 = approx $1,171.66.", "Compound interest formula"),
        ("A rectangular prism has length 12, width 5, and height 4. What is the length of the internal 3D space diagonal?", "Mathematics", "Diagonal = sqrt(12^2 + 5^2 + 4^2) = sqrt(144 + 25 + 16) = sqrt(185) approx 13.60.", "Pythagorean 3D theorem"),
        ("Compute the dot product of vectors u = [2, -3, 5] and v = [4, 1, -2].", "Mathematics", "u . v = (2*4) + (-3*1) + (5*-2) = 8 - 3 - 10 = -5.", "Vector algebra"),
        ("A bag contains 5 red, 4 blue, and 3 green marbles. Two marbles are drawn without replacement; find probability both are red.", "Mathematics", "(5/12) * (4/11) = 20/132 = 5/33 (approx 15.15%).", "Conditional probability"),
        ("What is the slope of the tangent line to the curve y = 2x^2 - 5x + 3 at the point where x = 3?", "Mathematics", "dy/dx = 4x - 5. At x = 3, slope = 4(3) - 5 = 7.", "Calculus slope calculation"),
        ("Find the roots of the quadratic equation 2x^2 - 7x + 3 = 0 using the quadratic formula.", "Mathematics", "x = (7 +/- sqrt(49 - 24))/4 = (7 +/- 5)/4. Roots are x = 3 and x = 0.5.", "Quadratic formula solution"),
        ("If a light year is approximately 9.46 trillion km, convert 4.2 light years into kilometers using scientific notation.", "Physics", "4.2 * 9.46 * 10^12 = 3.9732 * 10^13 kilometers.", "Scientific notation conversion"),
        ("Calculate the simple interest on a loan of $5,000 at an annual rate of 6.5% over 3 years.", "Finance", "I = P * r * t = 5000 * 0.065 * 3 = $975.", "Simple interest formula"),
        ("Find the perimeter of a right triangle whose legs are 9 cm and 12 cm.", "Mathematics", "Hypotenuse = sqrt(81 + 144) = 15 cm. Perimeter = 9 + 12 + 15 = 36 cm.", "Triangle geometry"),
        ("Calculate the angle between vectors [1, 0] and [1, 1] in degrees.", "Mathematics", "cos(theta) = 1 / sqrt(2), theta = 45 degrees.", "Trigonometric vector angle"),
        ("Evaluate the limit as x approaches 0 of sin(3x) / x.", "Mathematics", "Using L'Hopital or standard limit, 3 * cos(3x)/1 = 3.", "Calculus limit evaluation"),
        ("How many unique 4-letter permutations can be formed from the letters in the word 'PYTHON'?", "Mathematics", "6P4 = 6! / (6-4)! = 720 / 2 = 360 permutations.", "Combinatorial permutation"),

        # Science & Economics Analysis (45)
        ("Summarize the primary systemic causes of the 2008 global financial crisis into three distinct points.", "Economics", "1. Subprime mortgage deregulation. 2. Opaque credit derivatives (CDOs). 3. High leverage among investment banks.", "Structured economic summary"),
        ("Compare nuclear fission power generation with solar photovoltaic energy regarding carbon emissions, base-load capacity, and waste.", "Environmental Science", "Nuclear provides continuous base-load with high energy density and radioactive waste; solar is intermittent but produces zero operational emissions.", "Multi-attribute comparison"),
        ("Explain how mRNA vaccines stimulate antibody production without causing pathogenic viral replication.", "Biology", "They deliver mRNA encoding the viral spike protein, triggering ribosomic translation and antibody response without replicating live virus.", "Biomedical explanation"),
        ("Compare Keynesian stimulus policies with classical supply-side economics during sustained recessions.", "Economics", "Keynesians stimulate aggregate demand through government deficit spending; supply-siders cut taxes and regulation to boost supply.", "Macroeconomic synthesis"),
        ("Explain the monetary transmission mechanism by which central bank interest rate hikes reduce inflation.", "Economics", "Higher benchmark rates raise lending costs, lowering consumer borrowing and corporate capital investment, cooling demand.", "Economic causation"),
        ("How do the light-dependent reactions of photosynthesis generate ATP and NADPH for the Calvin cycle?", "Biology", "Photons excite chlorophyll electrons through photosystems II and I, powering proton pumps across thylakoid membranes to synthesize ATP and NADPH.", "Biochemical pathway"),
        ("Explain how negative feedback loops maintain homeostatic core temperature regulation in humans.", "Biology", "Thermoreceptors detect temperature shifts, prompting hypothalamus responses: vasodilation/sweating when hot, vasoconstriction/shivering when cold.", "Physiological regulation"),
        ("Compare the aerodynamic principles of Bernoulli's theorem with Newton's third law in generating aircraft wing lift.", "Physics", "Bernoulli explains low pressure over curved upper surfaces; Newton explains downward air deflection producing an equal upward reaction force.", "Aerodynamic comparison"),
        ("Explain how tariffs affect imported goods prices, domestic manufacturer competitiveness, and consumer welfare.", "Economics", "Tariffs raise import prices, shielding domestic producers from competition but driving up consumer costs and reducing real welfare.", "Trade policy analysis"),
        ("Describe the differences between an isothermal process and an adiabatic process in classical thermodynamics.", "Physics", "An isothermal process maintains constant temperature with heat transfer; an adiabatic process occurs without heat exchange with surroundings.", "Thermodynamic distinction"),
        ("Explain how enzyme catalytic rates are influenced by substrate concentration according to Michaelis-Menten kinetics.", "Biochemistry", "Velocity increases hyperbolically until active sites become saturated, asymptotically approaching maximum velocity Vmax.", "Biochemical kinetics"),
        ("Compare comparative advantage with absolute advantage in international trade theory.", "Economics", "Absolute advantage means producing goods with fewer inputs; comparative advantage means producing at lower opportunity cost.", "Economic trade theory"),
        ("How do geological subduction zones cause deep-focus earthquakes and explosive volcanic island arcs?", "Geology", "Dense oceanic plates sink beneath continental lithosphere, melting through hydration flux to produce explosive andesitic magma.", "Geological process"),
        ("Explain the electrochemical reaction occurring at the anode and cathode of a discharging lithium-ion battery.", "Chemistry", "Anode: Lithium deintercalates and oxidizes into Li+ and electrons. Cathode: Li+ ions and electrons intercalate into transition metal oxide.", "Electrochemical synthesis"),
        ("What causes the Coriolis effect and how does it steer global cyclone wind patterns in both hemispheres?", "Earth Science", "Earth's rotational velocity differences across latitudes deflect winds rightward in the Northern Hemisphere and leftward in the Southern.", "Atmospheric dynamics"),
        ("Compare monetary policy tools: open market operations, the discount rate, and reserve requirements.", "Economics", "OMOs buy/sell bonds to influence interbank reserves; discount rate sets direct Fed lending rates; reserve requirements dictate mandatory ratios.", "Central banking tools"),
        ("Explain the cellular mechanism of action of penicillin on bacterial cell wall synthesis.", "Pharmacology", "Penicillin binds to transpeptidase enzymes, preventing peptidoglycan cross-linking and causing osmotic bacterial lysis.", "Pharmacological mechanism"),
        ("Describe the differences between type 1 and type 2 diabetes mellitus regarding etiology and insulin dynamics.", "Medicine", "Type 1 is autoimmune destruction of pancreatic beta cells (insulin deficiency); Type 2 is peripheral insulin resistance with relative deficiency.", "Clinical etiology"),
        ("How does a mass spectrometer separate chemical isotopes based on their mass-to-charge ratio?", "Chemistry", "Sample molecules are ionized, accelerated into a magnetic field, and deflected along curved trajectories proportional to m/z.", "Instrumentation analysis"),
        ("Explain why monopolies produce lower output and charge higher prices than competitive markets.", "Economics", "A monopoly faces downward-sloping demand and sets marginal revenue equal to marginal cost, creating deadweight loss and economic rent.", "Microeconomic analysis"),
        ("Describe the life cycle of a massive star ending in a core-collapse supernova versus a white dwarf.", "Astrophysics", "Stars above 8 solar masses fuse elements up to iron before iron photodisintegration causes core collapse; low-mass stars shed envelopes into white dwarfs.", "Stellar evolution"),
        ("Explain the principles of chromatography and how stationary and mobile phases separate chemical mixtures.", "Chemistry", "Components partition differentially between the moving mobile phase and fixed stationary phase based on polarity and affinity.", "Chemical separation"),
        ("How does the human eye accommodate vision when shifting focus from a distant horizon to reading text?", "Physiology", "Ciliary muscles contract, releasing tension on zonular fibers, allowing the crystalline lens to thicken and increase refractive power.", "Sensory physiology"),
        ("Explain the economic concept of externalities and compare Pigouvian taxes with cap-and-trade systems.", "Economics", "Externalities are unpriced spillover costs/benefits. Pigouvian taxes price the negative externality per unit; cap-and-trade caps total emissions.", "Policy comparison"),
        ("Describe how the Doppler effect alters observed wave frequencies for moving wave sources.", "Physics", "Approaching sources compress wavefronts toward the observer (higher frequency); receding sources stretch wavefronts (lower frequency).", "Wave mechanics"),
        ("Explain the distinction between structural, frictional, and cyclical unemployment in macroeconomics.", "Economics", "Frictional is routine career transition; structural is skill/geographic mismatch; cyclical arises from macroeconomic downturns.", "Labor economics"),
        ("How does the greenhouse gas methane compare to carbon dioxide in terms of global warming potential and atmospheric lifespan?", "Environmental Science", "Methane has ~28-36x higher warming potential over 100 years than CO2, but oxidizes in the atmosphere within 12 years compared to centuries for CO2.", "Climate comparison"),
        ("Explain how CRISPR-Cas9 performs targeted genomic editing using guide RNA and endonuclease cleavage.", "Molecular Biology", "Guide RNA matches a target genomic locus adjacent to a PAM sequence, directing Cas9 endonuclease to introduce a double-strand break.", "Biotechnology mechanism"),
        ("Describe the differences between forward contracts and futures contracts in financial derivatives markets.", "Finance", "Forwards are customized over-the-counter contracts with counterparty risk; futures are standardized, exchange-traded, and marked-to-market daily.", "Derivative instruments"),
        ("Explain how the human auditory system transduces sound pressure waves in the cochlea into neural action potentials.", "Neuroscience", "Sound vibrates the oval window, generating fluid waves that displace the basilar membrane, bending hair cell stereocilia to open ion channels.", "Sensory transduction"),
        ("Compare fiscal austerity with debt-financed Keynesian stimulus during economic stagnation.", "Economics", "Austerity aims to reduce public debt but risks depressing demand; Keynesian stimulus borrows to boost demand at the risk of higher debt ratios.", "Macro policy debate"),
        ("Explain the difference between inductive and deductive reasoning with scientific examples of each.", "Philosophy of Science", "Deductive reasoning moves from general premises to guaranteed specific conclusions; inductive reasoning generalizes from specific observations.", "Epistemological reasoning"),
        ("Describe the physiological responses triggered by the sympathetic nervous system's fight-or-flight reaction.", "Physiology", "Releases epinephrine and norepinephrine, dilating bronchioles, increasing heart rate, inhibiting digestion, and mobilizing glucose.", "Autonomic response"),
        ("How does optical fiber exploit total internal reflection to transmit laser pulses across continents?", "Physics", "Light pulses enter a core with higher refractive index than cladding at angles exceeding the critical angle, reflecting along the core without escape.", "Optics application"),
        ("Explain the concept of liquidity trap in monetary economics and why conventional rate cuts become ineffective.", "Economics", "Interest rates reach near-zero, and demand for money becomes horizontal; injections of cash are hoarded rather than spent or invested.", "Macroeconomic theory"),
        ("Compare covalent bonding and ionic bonding regarding electron distribution and material properties.", "Chemistry", "Covalent bonds share electron pairs between non-metals; ionic bonds transfer electrons between metals and non-metals forming crystal lattices.", "Chemical bonding"),
        ("Explain how hydraulic systems amplify force using Pascal's principle.", "Physics", "Pressure applied to an enclosed incompressible fluid is transmitted undiminished; a smaller piston area generates a larger force on a larger piston area.", "Fluid mechanics"),
        ("Describe the stages of the business cycle and how GDP, employment, and inflation behave during each.", "Economics", "Expansion (rising GDP, low unemployment, rising inflation), Peak (capacity limits), Contraction (falling GDP, rising unemployment), Trough (bottom).", "Macroeconomic cycles"),
        ("How does renal filtration and reabsorption in nephrons regulate blood volume and electrolyte balance?", "Physiology", "Glomeruli filter plasma; proximal/distal tubules reabsorb water and electrolytes via aldosterone and ADH to regulate blood volume.", "Renal physiology"),
        ("Explain the difference between economic profit and accounting profit.", "Economics", "Accounting profit is revenue minus explicit monetary expenses; economic profit deducts both explicit costs and implicit opportunity costs.", "Microeconomic concepts"),
        ("How does an electrical transformer step up or step down AC voltage using electromagnetic induction?", "Physics", "Alternating current in the primary winding creates a time-varying magnetic flux in the core, inducing voltage in secondary windings proportional to turns ratio.", "Electromagnetism"),
        ("Explain the role of mycorrhizal fungi in terrestrial plant nutrient uptake and soil ecosystems.", "Ecology", "Mycorrhizae extend root surface area through hyphae, transferring phosphorus and minerals to plants in exchange for photosynthetic carbohydrates.", "Ecological symbiosis"),
        ("Compare open-market operations with quantitative easing when nominal interest rates reach the zero lower bound.", "Economics", "Conventional open market operations target short-term interbank rates; QE directly purchases long-term bonds to lower long-term borrowing yields.", "Monetary policy analysis"),
        ("Explain how plate tectonics drives the oceanic conveyor belt (thermohaline circulation).", "Oceanography", "Temperature and salinity differences drive deep water sinking in the North Atlantic, propelling global currents modulated by continental geometries.", "Oceanic systems"),
        ("Describe how a refrigeration cycle uses compression, condensation, expansion, and evaporation to transfer heat.", "Engineering", "Compressor heats refrigerant vapor; condenser rejects heat to surroundings; expansion valve drops pressure/temperature; evaporator absorbs interior heat.", "Thermodynamic cycles")
    ]

    # 100 Distinct Hard Queries
    hard = [
        # Advanced CS, Automata & Distributed Systems (30)
        ("Implement Manacher's algorithm in Python for linear O(n) palindromic substring discovery, detailing center expansion symmetry rules.", "Computer Science", "Manacher's algorithm maps strings into delimited forms and tracks palindrome boundaries to achieve O(n) worst-case complexity.", "Algorithmic engineering"),
        ("Prove by mathematical induction that 1^3 + 2^3 + ... + n^3 = [n(n+1)/2]^2 for all positive integers n.", "Mathematics", "Induction proof showing base case n=1 and inductive hypothesis step expanding (k+1)^3 into squared sum formulation.", "Formal mathematical proof"),
        ("Architect an end-to-end globally distributed rate limiter handling 500,000 req/sec across five continents with low latency.", "System Design", "Multi-region architecture using sliding window counter with Redis clusters, local token memory caching, and gossip protocol synchronization.", "Distributed systems architecture"),
        ("Solve the non-homogeneous second-order differential equation y'' - 4y' + 4y = e^(2x) using undetermined coefficients.", "Mathematics", "Characteristic equation has double root r=2; particular solution requires multiplying by x^2, yielding y = (c1 + c2*x + 0.5*x^2)e^(2x).", "Higher-order calculus"),
        ("Analyze the Raft consensus protocol: explain how log replication quorums mathematically prevent split-brain during network partitions.", "Distributed Systems", "Raft enforces strict majority (N/2 + 1) quorums for leader election and commitment, ensuring two disjoint partitions cannot commit conflicting logs.", "Distributed consensus proof"),
        ("Design a lock-free multi-producer multi-consumer (MPMC) ring buffer in C++ using atomic CAS and acquire-release semantics.", "Concurrent Systems", "Circular array buffer using atomic head/tail pointers, sequence counters, and std::memory_order_acquire/release to prevent torn reads.", "Concurrent systems programming"),
        ("Derive the backpropagation gradient updates for a neural network with softmax activation and cross-entropy loss from first principles.", "Machine Learning", "Derivation showing simplification where the Jacobian matrix diagonal and off-diagonal terms collapse neatly into (p - y).", "Mathematical ML derivation"),
        ("Explain how speculative execution vulnerabilities like Spectre exploit branch target buffers and CPU cache timing channels.", "Cybersecurity", "Attacker trains branch predictors to speculatively execute bounds-exceeding instructions, modifying cache state to leak data via flush+reload.", "Hardware security analysis"),
        ("Formulate a complete proof of the Cook-Levin Theorem establishing that the Boolean Satisfiability Problem (SAT) is NP-complete.", "Theoretical CS", "Encodes nondeterministic Turing machine configurations, transitions, and tape heads into polynomial-sized CNF formulas.", "Automata and complexity theory"),
        ("Synthesize a zero-downtime database migration architecture from a multi-terabyte monolithic relational database to a horizontally sharded Cassandra cluster.", "Data Architecture", "Dual-write pipeline utilizing CDC (Kafka/Debezium), parallel backfills, reconciliation checksumming, and gradual read traffic shifting.", "High-scale migration architecture"),
        ("Explain the construction of LALR(1) parsing tables from LR(0) item sets and how lookahead propagation resolves shift-reduce conflicts.", "Compiler Theory", "Merges LR(1) states sharing identical LR(0) cores, propagating lookaheads while avoiding introduced shift-reduce conflicts.", "Compiler construction"),
        ("Derive the time and space complexity of the Fast Fourier Transform (FFT) Cooley-Tukey algorithm using the Master Theorem.", "Computer Science", "Recurrence T(n) = 2T(n/2) + O(n). By Master Theorem case 2, log_b(a) = log_2(2) = 1 = d, yielding O(n log n).", "Algorithmic complexity derivation"),
        ("Formulate the Bellman-Ford shortest path algorithm with negative edge weights and prove its negative cycle detection criteria.", "Computer Science", "Relaxes all |V|-1 edges; if an |V|-th relaxation yields a shorter distance, a reachable negative cycle is mathematically proven to exist.", "Graph algorithm analysis"),
        ("Explain how Google Spanner uses TrueTime API with GPS and atomic clocks to enforce strict serializability across data centers.", "Distributed Systems", "TrueTime returns bounded time intervals [earliest, latest]; Spanner waits out the clock skew uncertainty window before committing writes.", "Distributed transactions"),
        ("Prove that the Halting Problem is undecidable using Cantor's diagonalization argument.", "Theoretical CS", "Constructs a hypothetical decider H, creates an adversarial program D that reverses H's prediction on itself, deriving an irreconcilable contradiction.", "Formal computability proof"),
        ("Design an optimal cache eviction policy (ARC: Adaptive Replacement Cache) dynamically balancing recency and frequency caches.", "Operating Systems", "Maintains dual doubly linked lists (L1 for recency, L2 for frequency) with ghost history lists to dynamically tune target size p.", "Cache algorithm design"),
        ("Derive the update equations for a Kalman filter estimating state and covariance matrices for a non-linear dynamic system (EKF).", "Robotics / Control", "Linearizes state transitions and measurement equations via Jacobian matrices, predicting prior state and updating via Kalman gain.", "Control theory mathematics"),
        ("Analyze Byzantine Fault Tolerance (PBFT): detail the pre-prepare, prepare, and commit phases and prove safety up to 3f + 1 nodes.", "Distributed Systems", "Requires 2f + 1 prepare messages to reach prepared certificate, and 2f + 1 commit messages to commit, ensuring agreement despite f malicious nodes.", "Byzantine consensus proof"),
        ("Formulate the A* heuristic search algorithm and prove that admissibility and consistency guarantee an optimal shortest path.", "Artificial Intelligence", "Proof by contradiction showing that if an optimal path exists, consistent heuristics f(n) = g(n) + h(n) expand nodes in monotonically non-decreasing order.", "Search algorithm proof"),
        ("Design a robust distributed tracing system similar to Dapper or OpenTelemetry handling 10 million spans per second with low overhead.", "Distributed Systems", "Injects trace and span IDs in RPC metadata headers, employs coherent head-based or tail-based sampling, and offloads spans via local daemon buffers.", "Observability systems architecture"),
        ("Explain how zero-knowledge SNARKs use polynomial commitment schemes and elliptic curve pairings to verify computations in O(1) time.", "Cryptography", "Converts computation to R1CS, polynomials via QAP, uses KZG commitments evaluated at secret points using toxic-waste pairing checks.", "Advanced cryptographic proof"),
        ("Prove that no comparison-based sorting algorithm can achieve an average-case time complexity better than Omega(n log n).", "Computer Science", "Decision tree argument: sorting n items requires at least n! leaves; a binary tree of height h has at most 2^h leaves; h >= log2(n!) = Omega(n log n).", "Information theoretic proof"),
        ("Design an LSM-tree (Log-Structured Merge-tree) storage engine detailing write ahead log, memtable, SSTables, and compaction strategies.", "Database Internals", "Appends writes to WAL and mutable memtable (skiplist); flushes immutable memtables to Levelled SSTables with Bloom filters and background compaction.", "Database internals design"),
        ("Formulate the mathematical formulation of the Viterbi algorithm for decoding hidden Markov model state sequences.", "Machine Learning", "Dynamic programming recurrence maintaining maximum probability path v_t(j) = max_i [v_{t-1}(i) * a_ij] * b_j(y_t) and backpointers.", "Probabilistic sequence modeling"),
        ("Analyze memory consistency models: compare x86 TSO (Total Store Order) with ARM relaxed consistency and explain store buffer forwarding.", "Computer Architecture", "x86 TSO allows reads to bypass older writes to different addresses via store buffers; ARM allows arbitrary reordering without explicit memory barriers.", "Microarchitectural consistency"),
        ("Derive the gradient equations for an Adam optimizer detailing bias correction for first and second moment moving averages.", "Machine Learning", "Derivation showing m_t/(1 - beta1^t) and v_t/(1 - beta2^t) correcting zero-initialization bias in early iteration timesteps.", "Mathematical optimization derivation"),
        ("Prove the existence of a stable matching for any marriage problem instance using the Gale-Shapley Deferred Acceptance algorithm.", "Game Theory", "Proof showing algorithm terminates in at most n^2 steps with men progressing down preferences and women improving matches monotonically.", "Combinatorial game theory"),
        ("Architect an event-driven CQRS and Event Sourcing architecture for high-frequency financial ledgers with snapshotting.", "System Architecture", "Separates command mutation from read views, storing immutable append-only event streams in Kafka/EventStore with periodic state snapshots.", "Enterprise financial architecture"),
        ("Explain how garbage collection in the Java HotSpot ZGC achieves sub-millisecond pause times using colored pointers and load barriers.", "Virtual Machines", "ZGC uses reference coloring in pointer bits and concurrent load barriers to relocate heap objects concurrently with application threads.", "Low-latency runtime internals"),
        ("Derive the recurrence relations for the Needleman-Wunsch global sequence alignment algorithm with affine gap penalties.", "Bioinformatics", "Three-matrix DP (M, Ix, Iy) separating match scores, gap opening penalties d, and gap extension penalties e to optimize in O(n*m).", "Computational biology algorithms"),

        # Advanced Mathematics, Physics & Quantitative Modeling (35)
        ("Derive the Black-Scholes-Merton partial differential equation from a delta-neutral hedging portfolio using Ito's Lemma.", "Quantitative Finance", "Constructs portfolio of long stock and short option; applies Ito's Lemma to option price dV, eliminating stochastic dW terms to derive PDE.", "Stochastic calculus derivation"),
        ("Derive the Euler-Lagrange equations from the Principle of Stationary Action using the calculus of variations.", "Theoretical Physics", "Action S = integral of L(q, q_dot, t) dt. Perturbs path by epsilon * eta(t), integrates by parts with boundary conditions vanishing, yielding EL equations.", "Variational mechanics derivation"),
        ("Prove the Central Limit Theorem using characteristic functions (Fourier transforms) and Taylor series expansions.", "Statistics", "Characteristic function of standardized sum converges pointwise to exp(-t^2 / 2), which is the unique characteristic function of standard normal.", "Probability theory proof"),
        ("Derive the Maxwell-Boltzmann velocity distribution from the canonical ensemble partition function in statistical thermodynamics.", "Statistical Mechanics", "Integrates phase space volume over momentum coordinates using Gaussian integrals to evaluate the single-particle partition function.", "Statistical mechanics proof"),
        ("Formulate the Navier-Stokes equations for incompressible Newtonian fluid flow and explain the millennium prize open problem.", "Fluid Dynamics", "Applies Newton's second law to continuum elements; open problem asks whether smooth, physically reasonable solutions always exist in 3D for all time.", "Mathematical physics"),
        ("Derive the Runge-Kutta 4th-order (RK4) numerical integration formula using Taylor series expansions and Simpson's 1/3 rule.", "Numerical Analysis", "Expands f(x, y) along intermediate slope estimates k1, k2, k3, k4, matching Taylor series expansion coefficients up to order O(h^4).", "Numerical differential equations"),
        ("Prove the spectral theorem for finite-dimensional Hermitian operators and show their eigenvalues are strictly real.", "Linear Algebra", "Proof showing conjugate transpose equality implies lambda * <v, v> = lambda_bar * <v, v>, proving lambda is real and eigenvectors are orthogonal.", "Linear algebra proof"),
        ("Derive the Schrodinger wave equation for a quantum particle in a 1D finite potential well, detailing boundary condition transcendental equations.", "Quantum Mechanics", "Solves piecewise Helmholtz equations inside and outside well; matches wavefunctions and derivatives at boundaries to yield transcendental energy equations.", "Quantum mechanics derivation"),
        ("Derive the pricing formula for a European call option using risk-neutral valuation and Gaussian cumulative integrals.", "Quantitative Finance", "Expectation of max(S_T - K, 0) under risk-neutral measure Q, substituting log-normal integration limits to yield S*N(d1) - K*e^(-rT)*N(d2).", "Risk-neutral pricing derivation"),
        ("Prove that the Mandelbrot set is connected using Douady and Hubbard's uniformization of its exterior complement.", "Complex Analysis", "Constructs conformal isomorphisms from the complement of the Mandelbrot set to the exterior of the unit disk via Green's functions.", "Complex dynamics proof"),
        ("Derive Einstein's field equations from the Einstein-Hilbert action using the Palatini variational method in general relativity.", "General Relativity", "Varies metric tensor g^(mu nu) in action S = integral (R - 2*Lambda) sqrt(-g) d4x, applying Bianchi identities to obtain G_{mu nu} = 8*pi*T_{mu nu}.", "Relativistic field derivation"),
        ("Formulate the Karush-Kuhn-Tucker (KKT) necessary and sufficient conditions for non-linear convex optimization with inequality constraints.", "Optimization Theory", "Primal feasibility, dual feasibility, stationarity of the Lagrangian, and complementary slackness conditions for global optima.", "Convex optimization theory"),
        ("Derive the Gibbs-Duhem equation from the fundamental thermodynamic relation and Euler's homogeneous function theorem.", "Thermodynamics", "Applies Euler's theorem to internal energy U(S, V, N), differentiates total differential, and subtracts to yield S*dT - V*dP + sum(N_i * dmu_i) = 0.", "Thermodynamic derivation"),
        ("Prove the Brouwer fixed-point theorem for the unit disk using algebraic topology and the non-retraction theorem.", "Topology", "Assumes fixed-point-free map f, constructs smooth retraction from disk onto boundary circle, showing induced homology group contradiction Z -> 0.", "Algebraic topology proof"),
        ("Derive the Merton jump-diffusion asset pricing model equations incorporating compound Poisson process shocks.", "Financial Engineering", "Extends Black-Scholes geometric Brownian motion with log-normal jump sizes arriving via Poisson process rate lambda, deriving integro-differential PDE.", "Quantitative jump modeling"),
        ("Formulate the Dirichlet problem for Poisson's equation on a sphere using spherical harmonics and Green's functions.", "Mathematical Physics", "Expands potential in Legendre polynomials and spherical harmonics Y_lm(theta, phi), matching boundary charges with orthogonality relations.", "Partial differential equations"),
        ("Derive the maximum entropy probability distribution for a continuous random variable given mean and variance constraints.", "Information Theory", "Maximizes differential entropy -integral p(x) ln p(x) dx subject to normalization and moment constraints via calculus of variations, yielding Gaussian.", "Information theory derivation"),
        ("Prove that every finite field has prime power order q = p^k and that its multiplicative group is cyclic.", "Abstract Algebra", "Shows characteristic must be prime p; vector space over Z_p gives size p^k; uses polynomial roots and Euler totient to prove cyclic structure.", "Abstract algebra proof"),
        ("Derive the Lorentz transformation equations from Einstein's two postulates of special relativity.", "Relativity", "Assumes linear spacetime transformation; applies constancy of speed of light in all inertial frames to determine gamma factor sqrt(1 - v^2/c^2).", "Theoretical physics derivation"),
        ("Formulate the Fokker-Planck equation describing the time evolution of the probability density function of an Ito drift-diffusion process.", "Stochastic Processes", "Applies Kramers-Moyal forward expansion to the Chapman-Kolmogorov equation, retaining drift and diffusion second-derivative terms.", "Stochastic physics"),
        ("Prove the Cauchy-Schwarz inequality for general inner product spaces and state conditions for equality.", "Functional Analysis", "Considers quadratic polynomial in t: ||u + t*v||^2 >= 0; discriminant must be <= 0, proving |<u, v>|^2 <= <u, u>*<v, v> with equality iff linear dependence.", "Functional analysis proof"),
        ("Derive the thermodynamic efficiency of an ideal Carnot cycle operating between temperatures Th and Tc from entropy balance.", "Thermodynamics", "Entropy change across reversible cycle is zero; Q_h/T_h = Q_c/T_c; efficiency = (W/Q_h) = (Q_h - Q_c)/Q_h = 1 - T_c/T_h.", "Classical thermodynamics derivation"),
        ("Formulate the mathematical proof of the Nyquist-Shannon sampling theorem using Dirac comb convolution and Fourier transforms.", "Signal Processing", "Multiplies time signal by Dirac comb; Fourier transform becomes periodic spectrum copies; bandlimited signals avoid aliasing if fs > 2B.", "Signal processing proof"),
        ("Derive the geodesic equation of motion from the variational principle of spacetime proper time in curved geometry.", "General Relativity", "Varies proper time delta integral ds = 0; uses metric connection and Christoffel symbols Gamma^lambda_{mu nu} to derive acceleration equation.", "Differential geometry derivation"),
        ("Prove that the Riemann zeta function satisfies the functional equation relating zeta(s) to zeta(1-s) using Jacobi theta functions.", "Analytic Number Theory", "Uses Mellin transform of Jacobi theta function, Poisson summation formula, and contour integration over the critical strip.", "Analytic number theory proof"),
        ("Derive the capital asset pricing model (CAPM) equation from the Markowitz mean-variance efficient frontier and tangency portfolio.", "Quantitative Economics", "Maximizes Sharpe ratio to find tangency portfolio; applies first-order conditions to show asset risk premium equals beta times market risk premium.", "Financial economics derivation"),
        ("Formulate the Navier-Stokes boundary layer equations (Prandtl's equations) for high Reynolds number flow past a flat plate.", "Aerodynamics", "Applies order-of-magnitude scaling analysis to discard small vertical pressure gradients, simplifying momentum equations into parabolic form.", "Fluid boundary layer theory"),
        ("Prove that every open set in R^n can be expressed as a countable union of disjoint open intervals/cubes.", "Real Analysis", "Uses dyadic cubes of side length 2^(-k); selects maximal disjoint dyadic cubes interior to the open set to form countable partition.", "Real analysis proof"),
        ("Derive the Stefan-Boltzmann law E = sigma * T^4 by integrating Planck's blackbody radiation formula over all frequencies.", "Thermodynamics", "Integrates u(nu) dnu over (0, infinity); substitutes dimensionless variable x = h*nu / (k*T), using Riemann zeta integral integral x^3/(e^x - 1) = pi^4 / 15.", "Blackbody radiation derivation"),
        ("Formulate the non-linear Schrodinger equation (NLSE) modeling optical soliton propagation in non-linear Kerr media.", "Non-Linear Optics", "Combines group velocity dispersion with self-phase modulation in optical fiber to derive sech(t) shape-preserving bright solitons.", "Non-linear optics derivation"),
        ("Prove that the set of algebraic numbers is countable while the set of transcendental numbers is uncountably infinite.", "Set Theory", "Polynomials of degree n with integer coefficients are countable; countable union of countable sets is countable; R is uncountable, so transcendentals are uncountable.", "Set theory proof"),
        ("Derive the Fermi-Dirac and Bose-Einstein quantum distribution statistics from the grand canonical ensemble.", "Quantum Statistics", "Grand partition function with occupation numbers restricted to 0, 1 (fermions) or 0 to infinity (bosons), differentiating ln(Xi) with respect to chemical potential.", "Quantum statistical mechanics"),
        ("Formulate the variational formulation (weak form) for a Poisson boundary value problem and prove solution existence via Lax-Milgram theorem.", "Numerical PDE", "Multiplies PDE by test function, integrates by parts; shows bilinear form is continuous and coercive in Sobolev space H^1_0, guaranteeing unique weak solution.", "Sobolev space functional analysis"),
        ("Derive the Gibbs free energy minimization criteria for multi-component chemical equilibrium with stoichiometric constraints.", "Chemical Thermodynamics", "Applies Lagrange multipliers to chemical potentials sum(nu_i * mu_i) = 0, deriving the equilibrium constant K_p relationship with standard delta G.", "Chemical thermodynamics derivation"),
        ("Prove the Picard-Lindelof theorem guaranteeing local existence and uniqueness of solutions to initial value ordinary differential equations.", "Differential Equations", "Converts ODE to integral equation; defines Picard iteration operator on Banach space; proves operator is contraction mapping under Lipschitz condition.", "ODE analysis proof"),

        # Advanced Engineering & Algorithmic Synthesis (35)
        ("Design an automated market maker (AMM) invariant bonding curve that minimizes impermanent loss for concentrated liquidity positions.", "FinTech / Cryptography", "Derives virtual reserve equations similar to Uniswap v3, calculating tick spacing, capital efficiency multipliers, and dynamic fee scaling.", "Algorithmic economics"),
        ("Synthesize a high-throughput lock-free priority queue using multi-dimensional skip lists and atomic pointer operations.", "Concurrent Systems", "Implements probabilistic skiplist with atomic compare-and-swap on forward node pointers, utilizing deletion flags to handle concurrent dequeues.", "Concurrent data structures"),
        ("Design a fault-tolerant geo-replicated distributed consensus state machine with dynamic membership reconfiguration.", "Distributed Systems", "Integrates joint consensus protocol to safely transition cluster configurations from C_old to C_new without split-quorum vulnerabilities.", "Distributed systems engineering"),
        ("Formulate the mathematics of deep reinforcement learning Proximal Policy Optimization (PPO) clipped surrogate objective functions.", "Machine Learning", "Derives objective function clipping probability ratio r_t(theta) within [1 - epsilon, 1 + epsilon], preventing destabilizingly large policy updates.", "Reinforcement learning theory"),
        ("Architect an ultra-low latency FPGA-accelerated financial market data processing feed handler analyzing packet parsing and PCIe DMA overhead.", "Hardware Systems", "Direct Ethernet MAC parser on FPGA parsing FIX/FAST packets, maintaining L1 orderbook in on-chip BRAM, dispatching DMA writes to host memory in <1us.", "Low-latency systems architecture"),
        ("Synthesize a formal security proof for zero-knowledge range proofs using Bulletproofs and inner product arguments.", "Cryptography", "Encodes commitment value in binary representation, reduces range checks to inner product argument over Pedersen commitments in logarithmic proof size.", "Cryptographic protocol proof"),
        ("Design an end-to-end autonomous driving trajectory planner combining hybrid A* pathfinding with non-linear Model Predictive Control (MPC).", "Robotics", "Hybrid A* search over continuous 3D vehicle state space (x, y, theta), feeding kinematically feasible trajectories into real-time quadratic programming MPC.", "Robotics control synthesis"),
        ("Formulate the mathematics of attention sink phenomenon and streaming LLM architectures for infinite context window inference.", "Machine Learning Theory", "Analyzes excessive attention softmax allocation to initial sequence tokens; preserves first four tokens alongside sliding window KV-cache to stabilize attention.", "LLM architectural theory"),
        ("Design a high-frequency algorithmic arbitrage pipeline across fragmented cryptocurrency decentralized exchanges with MEV protections.", "FinTech", "Simulates pending mempool bundles, evaluates flash loans, solves convex arbitrage routing graphs, and submits private RPC bundles to block builders.", "Algorithmic trading systems"),
        ("Prove the convergence of the Alternating Direction Method of Multipliers (ADMM) for separable convex optimization problems.", "Optimization Theory", "Proof showing residual convergence, objective convergence, and dual variable convergence via Lyapunov energy functions.", "Convex optimization proof"),
        ("Architect a high-availability distributed file system combining erasure coding (Reed-Solomon) and tiered NVMe caching.", "Storage Systems", "Implements (8+4) Reed-Solomon coding striping blocks across storage nodes, maintaining metadata servers with Raft and write-back NVMe caches.", "Distributed storage engineering"),
        ("Derive the backpropagation equations through time (BPTT) for an LSTM cell, detailing forget, input, output, and cell state gradients.", "Machine Learning", "Calculates recursive partial derivatives dC_t/dC_{t-1} across sequential unrolled timesteps, explaining how additive cell states mitigate vanishing gradients.", "Deep learning derivation"),
        ("Synthesize a Byzantine-fault-tolerant clock synchronization algorithm for heterogeneous distributed sensor networks.", "Distributed Computing", "Combines Marzullo's interval intersection algorithm with Lamport logical clocks, rejecting up to f aberrant sensor readings.", "Distributed sensor algorithms"),
        ("Formulate the theoretical foundations of generative adversarial network (GAN) training as a minimax game optimizing Jensen-Shannon divergence.", "Machine Learning", "Derives optimal discriminator D*(x) = p_data / (p_data + p_g); substitutes into generator objective to prove value function equals 2*JSD(p_data || p_g) - 2 ln 2.", "Generative modeling proof"),
        ("Design a compilation pass in LLVM for automated vectorization (SIMD) analyzing polyhedral loop dependency graphs.", "Compiler Design", "Models loop iterations as integer polyhedra, evaluates distance and direction vectors, and applies unimodular transformations for AVX-512 vectorization.", "Compiler optimization"),
        ("Synthesize an end-to-end multi-modal diffusion model architecture aligning continuous latent audio and visual representations.", "Generative AI", "Cross-attention conditioning between audio spectrogram latents and video frame latents with classifier-free guidance and temporal self-attention.", "Multi-modal model design"),
        ("Formulate the mathematical proof of the Cramér-Rao lower bound for unbiased parameter estimators in statistical estimation theory.", "Mathematical Statistics", "Applies Cauchy-Schwarz inequality to covariance between estimator and score function, proving variance cannot be less than inverse Fisher information.", "Statistical inference proof"),
        ("Architect a massively parallel graph neural network (GNN) training pipeline over trillion-edge graphs using partition-based sampling.", "Data Engineering", "Combines METIS graph partitioning with neighborhood sampling (GraphSAGE), overlapping GPU-CPU communications via NVLink and asynchronous CUDA streams.", "High-performance computing"),
        ("Derive the equations of motion for an inverted pendulum on a cart using Hamiltonian mechanics and design a linear quadratic regulator (LQR).", "Control Systems", "Derives Legendre transform to find Hamiltonian H(q, p); linearizes equations of motion around unstable upright equilibrium; solves Riccati equation for LQR gain.", "Control systems engineering"),
        ("Formulate the mathematical equations for quantum error correction using the surface code (toric code) and minimum-weight perfect matching.", "Quantum Computing", "Defines star and plaquette stabilizer operators on 2D qubit lattice, measuring syndrome eigenvalues to identify and correct bit and phase errors via Blossom algorithm.", "Quantum fault tolerance"),
        ("Design a zero-copy network packet processing pipeline in the Linux kernel using eBPF and XDP (eXpress Data Path).", "Systems Programming", "Attaches XDP programs directly to NIC driver ring buffers, rewriting Ethernet/IP headers and redirecting packets via AF_XDP sockets without socket buffer allocations.", "Kernel systems programming"),
        ("Synthesize a formal specification for a two-phase commit protocol using TLA+ and model check safety and liveness invariants.", "Formal Methods", "Defines transaction managers and resource managers states; specifies Prepare, Commit, Abort actions, verifying non-triviality and consensus under crash faults.", "Formal verification"),
        ("Formulate the mathematics of LoRA (Low-Rank Adaptation) and QLoRA for parameter-efficient fine-tuning of large language models.", "Machine Learning", "Decomposes weight updates Delta W as B * A where B in R^(d x r) and A in R^(r x k) with r << d; analyzes double quantization and 4-bit NormalFloat (NF4).", "LLM optimization theory"),
        ("Design a distributed multi-tenant vector database capable of indexing and querying 1 billion 1536-dimensional vectors with sub-50ms latency.", "Vector Search", "Combines Hierarchical Navigable Small World (HNSW) graphs with Product Quantization (PQ), inverted file indexes (IVF), and memory-mapped persistent storage.", "Vector database engineering"),
        ("Derive the mathematical formulation of contrastive predictive coding (CPC) and InfoNCE loss as a lower bound on mutual information.", "Unsupervised Learning", "Proves InfoNCE loss L_q >= I(X; C) - ln(N), showing that maximizing cosine similarity with positive targets relative to negative distractors maximizes mutual information.", "Representation learning proof"),
        ("Architect a deterministic zero-knowledge virtual machine (zkVM) execution trace proving arithmetic circuits and memory consistency.", "Cryptography", "Compiles RISC-V bytecode into execution matrices, proves memory state transitions using Permutation Arguments (Plookup) and STARK polynomial commitments.", "zkVM architecture"),
        ("Formulate the mathematics of continuous-time normalizing flows using Neural Ordinary Differential Equations (Neural ODEs).", "Machine Learning Theory", "Parameterizes velocity fields f(z(t), t) with neural networks; evaluates change in log probability via instantaneous change of variables trace formula.", "Generative neural theory"),
        ("Design an automated pipeline for formal software verification of smart contracts using symbolic execution and SMT solvers (Z3).", "Software Verification", "Translates EVM bytecode into control flow graphs, accumulates path conditions symbolically, and queries Z3 solver to detect reentrancy and integer overflows.", "Symbolic software verification"),
        ("Derive the mathematical equations for thermal noise (Johnson-Nyquist noise) and shot noise in semiconductor p-n junctions.", "Semiconductor Physics", "Applies fluctuation-dissipation theorem to calculate Johnson noise spectral density 4kTR; analyzes Poisson arrival of discrete carriers for shot noise 2qI.", "Solid state physics derivation"),
        ("Synthesize an end-to-end multi-agent orchestration framework implementing democratic consensus, debate protocols, and memory reflection.", "AI Agent Architecture", "Orchestrates multi-agent debate loops with role specializations, recursive self-reflection buffers, and weighted vote tallying to prevent hallucination cascades.", "Agentic systems engineering"),
        ("Formulate the mathematics of diffusion model score matching (Denoising Score Matching) and prove equivalence to continuous Langevin dynamics.", "Machine Learning", "Proves that minimizing expectation of ||s_theta(x_t, t) - grad_x log p(x_t | x_0)||^2 directly learns the score function of data distribution.", "Generative probability proof"),
        ("Architect a high-performance memory-mapped columnar analytical database engine in Rust utilizing SIMD vectorization and Apache Arrow.", "Database Engineering", "Zero-copy query execution over Arrow RecordBatches, generating vectorized AVX2 filter kernels and multi-threaded parallel hash aggregations via Rayon.", "Columnar database engineering"),
        ("Derive the relativistic hydrodynamic equations for a viscous quark-gluon plasma created in relativistic heavy-ion collisions.", "Nuclear Physics", "Expands energy-momentum tensor T^(mu nu) up to second order in dissipative gradients using Israel-Stewart relaxation equations.", "High energy physics derivation"),
        ("Formulate a complete proof of the PCP Theorem (Probabilistically Checkable Proofs) and its implications for hardness of approximation.", "Theoretical CS", "Proof outline showing NP = PCP(O(log n), O(1)), showing that verifying proofs with logarithmic random coins and constant queries proves MAX-3SAT hardness.", "Computational complexity proof"),
        ("Synthesize an end-to-end resilient microgrid energy management controller combining stochastic dynamic programming and battery degradation models.", "Power Systems Engineering", "Formulates Bellman value iterations accounting for probabilistic renewable generation forecasts, wholesale electricity spot prices, and battery lithium plating limits.", "Smart grid optimization")
    ]

    records = []
    
    # Process Easy (100)
    for i, item in enumerate(easy):
        q_text, domain, ref_ans, capability = item
        score = round(0.08 + (i % 20) * 0.010 + len(q_text) * 0.0004, 3)
        score = min(0.35, max(0.05, score))
        records.append({
            "query_id": f"Q{len(records)+1:03d}",
            "query": q_text,
            "category": "Easy",
            "difficulty_score": score,
            "domain": domain,
            "expected_capability": capability,
            "reference_answer": ref_ans
        })

    # Process Medium (100)
    for i, item in enumerate(medium):
        q_text, domain, ref_ans, capability = item
        score = round(0.38 + (i % 25) * 0.011 + len(q_text) * 0.0003, 3)
        score = min(0.69, max(0.36, score))
        records.append({
            "query_id": f"Q{len(records)+1:03d}",
            "query": q_text,
            "category": "Medium",
            "difficulty_score": score,
            "domain": domain,
            "expected_capability": capability,
            "reference_answer": ref_ans
        })

    # Process Hard (100)
    for i, item in enumerate(hard):
        q_text, domain, ref_ans, capability = item
        score = round(0.72 + (i % 25) * 0.010 + len(q_text) * 0.0002, 3)
        score = min(0.98, max(0.71, score))
        records.append({
            "query_id": f"Q{len(records)+1:03d}",
            "query": q_text,
            "category": "Hard",
            "difficulty_score": score,
            "domain": domain,
            "expected_capability": capability,
            "reference_answer": ref_ans
        })

    df = pd.DataFrame(records)
    return df, records

def save_and_validate_dataset():
    os.makedirs("data", exist_ok=True)
    df, records = build_unique_dataset()

    csv_path = "data/queries.csv"
    json_path = "data/queries.json"

    df.to_csv(csv_path, index=False, encoding="utf-8")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # Rigorous validation checks
    assert len(df) == 300, f"Expected 300 queries, got {len(df)}"
    counts = df["category"].value_counts().to_dict()
    assert counts.get("Easy") == 100, f"Expected 100 Easy queries, got {counts.get('Easy')}"
    assert counts.get("Medium") == 100, f"Expected 100 Medium queries, got {counts.get('Medium')}"
    assert counts.get("Hard") == 100, f"Expected 100 Hard queries, got {counts.get('Hard')}"
    assert df["query"].nunique() == 300, f"Expected 300 unique queries, got {df['query'].nunique()}"

    print(f"[OK] Generated {len(df)} benchmark queries.")
    print(f"Distribution: {counts}")
    print(f"Unique Queries: {df['query'].nunique()} / 300")
    print(f"Saved successfully to '{csv_path}' and '{json_path}'")

if __name__ == "__main__":
    save_and_validate_dataset()
