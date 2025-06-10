# 🚀 Spaceship Titanic - Passenger Transport Classification
##  Objective
To predict whether a passenger was **transported to an alternate dimension** during the collision of Spaceship Titanic with a spacetime anomaly. This analysis involves preprocessing, exploratory analysis, visualization, dimensionality reduction, and machine learning modeling.

---

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

## 1. **Preprocessing**
   - Dropped `PassengerId`, `Cabin`, and `Name`
   - Extracted the `Deck` from the `Cabin` column
   - Imputed missing numerical features using median, and categorical using mode
  

## 2. **Exploratory Data Analysis (EDA)**

### Insight 1: Transport Rate by HomePlanet
| HomePlanet | Transport Rate |
|------------|----------------|
| Europa     | 65.88%         |
| Mars       | 52.30%         |
| Earth      | 42.39%         |

**Conclusion:** Passengers from **Europa** had the highest chance of being transported.

---

### Insight 2: Impact of CryoSleep
| CryoSleep | Transport Rate |
|-----------|----------------|
| True      | 81.76%         |
| False     | 32.89%         |

**Conclusion:** Passengers in **CryoSleep** were much more likely to be transported.

---

### Insight 3: VIP Status
| VIP Status | Transport Rate |
|------------|----------------|
| False      | 50.63%         |
| True       | 38.19%         |

**Conclusion:** **Non-VIP** passengers had a higher transport rate.

---

### Insight 4: Deck-wise Transport Rate (with Pie Charts)
Deck-wise transport percentages were visualized using pie charts.

Example insights:
- **Deck B** had the highest transport rate (~76%)
- **Deck F** had the lowest (~28%)

---

## 3. **Dimensionality Reduction**
   - Used **PCA** to reduce feature space to 2D
   - The scatter plot indicated some separation between transported and non-transported clusters.

## 4. **Modeling**
   - Still didn't finished.
---


📌 Libraries Used

    pandas

    numpy

    seaborn

    matplotlib

    sklearn (PCA, preprocessing, model selection, metrics)
