# ⚔️ JJK vs Demon Slayer: Ultimate War Simulation

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Anime](https://img.shields.io/badge/anime-crossover-ff69b4.svg)

A fun Python project that uses **Machine Learning** to predict epic battles between Jujutsu Kaisen and Demon Slayer characters! Watch Sukuna and Toji take on all the Upper Moon Demons and Hashira in an all-out war simulation.

## 🎯 What Does This Do?

This project simulates an epic crossover battle between:
- **JJK Team**: Sukuna (20 Fingers) & Toji Fushiguro
- **Demon Slayer Team**: All Upper Moons (Muzan, Kokushibo, Doma, Akaza, etc.) & All 9 Hashira

Using a **Random Forest Classifier**, the program predicts battle outcomes based on character stats like:
- Physical power
- Speed
- Durability
- Technique versatility
- Combat IQ
- Regeneration
- Special abilities (Domain Expansion, Blood Demon Arts, etc.)

## ✨ Features

- 🤖 **ML-Based Battle Predictor**: Trained on 3000+ synthetic matchups
- 📊 **Character Stats Database**: Detailed stats for 18+ characters from both series
- ⚔️ **Individual Matchups**: See who wins in 1v1 battles with win probabilities
- 🌍 **War Simulation**: Full-scale war scenarios with team power analysis
- 📈 **Statistical Analysis**: Win rates, power ratios, and battle narratives

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy pandas scikit-learn
```

### Run the Simulation

```bash
python jjk_vs_ds_full_war.py
```

## 📋 What You'll See

When you run the simulation, you'll get:

1. **Training Phase**: ML model trains on character matchups
2. **Phase 1**: Individual battle predictions (Sukuna vs Muzan, Toji vs Gyomei, etc.)
3. **Phase 2**: Team battle simulations with power level calculations
4. **Phase 3**: Final war outcome with detailed narrative
5. **Statistics**: Win records, average probabilities, and matchup breakdowns

### Sample Output

```
JJK vs DEMON SLAYER: ULTIMATE WAR SIMULATION
Sukuna & Toji vs Upper Moon Demons & Hashira
======================================================================

TRAINING MACHINE LEARNING BATTLE PREDICTOR
Training Accuracy: 0.875
Test Accuracy: 0.842

PHASE 1: KEY INDIVIDUAL MATCHUP PREDICTIONS
----------------------------------------------------------------------

Sukuna (20 Fingers) vs Muzan Kibutsuji
  Winner: Sukuna (20 Fingers) (High confidence)
  Win Probabilities: Sukuna: 87.3% | Muzan: 12.7%
  Power Scores: 162.5 vs 128.4
    - Sukuna's Domain Expansion creates overwhelming advantage
    - Sukuna's Cleave can bypass demon regeneration
```

## 🎮 Character Roster

### JJK Team (2)
- **Sukuna (20 Fingers)**: The King of Curses with Domain Expansion
- **Toji Fushiguro**: The Sorcerer Killer with Heavenly Restriction

### Demon Slayer Team (16)

**Upper Moons (7):**
- Muzan Kibutsuji
- Kokushibo (Upper Moon 1)
- Doma (Upper Moon 2)
- Akaza (Upper Moon 3)
- Hantengu (Upper Moon 4)
- Gyokko (Upper Moon 5)
- Gyutaro & Daki (Upper Moon 6)

**Hashira (9):**
- Gyomei Himejima (Stone)
- Sanemi Shinazugawa (Wind)
- Muichiro Tokito (Mist)
- Obanai Iguro (Serpent)
- Mitsuri Kanroji (Love)
- Giyu Tomioka (Water)
- Shinobu Kocho (Insect)
- Tengen Uzui (Sound)
- Kyojuro Rengoku (Flame)

## 🔧 How It Works

### 1. Character Database
Each character has detailed stats:
```python
"Sukuna (20 Fingers)": {
    "physical_power": 100,
    "speed": 98,
    "cursed_energy": 100,
    "domain_expansion": 100,
    "special_abilities": ["Malevolent Shrine", "Cleave", "Dismantle"]
}
```

### 2. Machine Learning Model
- **Algorithm**: Random Forest Classifier
- **Training Data**: 3000 synthetic battles based on stat differences
- **Features**: Power diff, speed diff, durability diff, regen diff, etc.
- **Output**: Win probability for each matchup

### 3. Matchup Modifiers
Special interactions between power systems:
- Domain Expansion gets +35% boost
- Cursed Energy vs Demons: +15%
- Toji vs Hashira: +15%
- Demon Regeneration: +12%

### 4. War Simulation
Three scenarios:
- JJK vs All Hashira
- JJK vs All Upper Moons
- JJK vs Everyone

## 🎲 Fun Stats

The simulation considers unique matchup dynamics:
- Sukuna's Domain Expansion overwhelms most opponents
- Toji dominates Hashira but struggles against demon regeneration
- Upper Moons' regeneration counters human attacks
- Power scaling differences between the two universes

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Feel free to:
- Add more characters
- Adjust character stats
- Improve the ML model
- Add visualization features
- Create different scenarios

## 💡 Ideas for Extension

- 🎨 Add visualization with matplotlib
- 🎮 Make it interactive (choose matchups)
- 📊 Add more detailed battle narratives
- 🌟 Include more JJK characters (Gojo, Yuta, etc.)
- 🔥 Add environmental factors (night time, sunlight, etc.)
- 🎯 Implement team composition strategies

## ⚠️ Disclaimer

This is a **fan project** for entertainment purposes. All characters and series belong to their respective creators:
- **Jujutsu Kaisen** by Gege Akutami
- **Demon Slayer** by Koyoharu Gotouge

Character power levels and matchup predictions are subjective and based on the creator's interpretation!

## 🎉 Enjoy!

Who do **you** think would win? Run the simulation and find out! 

---

**Made with ❤️ by an anime fan who loves data science**
