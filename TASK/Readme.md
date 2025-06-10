# 🚀 Spaceship Titanic - Passenger Transport Classification

## 📄 Description of the Task
- Explore and preprocess the data
- Generate and visualize insights
- Perform dimensionality reduction
- Build machine learning models to predict whether a passenger was transported

---

## 📊 Dataset Overview


| Column         | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| PassengerId    | Unique identifier for each passenger (grouped by ID prefix)                |
| HomePlanet     | Planet the passenger boarded from                                           |
| CryoSleep      | Whether the passenger was in cryo-sleep during the journey                 |
| Cabin          | Passenger's cabin (split into deck/num/side)                               |
| Destination    | Destination planet                                                         |
| Age            | Age of the passenger                                                       |
| VIP            | Whether the passenger paid for special VIP service                         |
| RoomService    | Amount spent on room service                                                |
| FoodCourt      | Amount spent in the food court                                              |
| ShoppingMall   | Amount spent in the shopping mall                                           |
| Spa            | Amount spent on spa services                                                |
| VRDeck         | Amount spent on virtual reality deck                                        |
| Name           | Full name of the passenger                                                  |
| Transported    | Target variable: whether the passenger was transported to another dimension|

---

## 🧠 Approach Followed

1. **Preprocessing**
   - Dropped `PassengerId`, `Cabin`, and `Name`
   - Extracted the `Deck` from the `Cabin` column
   - Imputed missing numerical features using median, and categorical using mode
  

2. **Exploratory Data Analysis**
   - Bar plots to explore relationships between features and `Transported`
   - Pie charts for transportation percentages across different `Deck`s
   - Age distribution plotted

3. **Dimensionality Reduction**
   - Used **PCA** to reduce feature space to 2D
   - Visualized the 2D PCA space with color-coded classes

4. **Modeling**
   - Still didn't finished.
---

📌 Libraries Used

    pandas

    numpy

    seaborn

    matplotlib

    sklearn (PCA, preprocessing, model selection, metrics)
