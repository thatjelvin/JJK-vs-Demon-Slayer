"""
JJK vs Demon Slayer: Full War Simulation
Sukuna & Toji vs Upper Moon Demons and Hashira
Machine Learning-Based Battle Predictor with War Simulation
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# CHARACTER DATABASE
# ============================================================================

# JJK Characters (Only Sukuna and Toji as per request)
JJK_TEAM = {
    "Sukuna (20 Fingers)": {
        "physical_power": 100,
        "speed": 98,
        "durability": 97,
        "cursed_energy": 100,
        "technique_versatility": 100,
        "combat_iq": 98,
        "regeneration": 95,
        "domain_expansion": 100,
        "special_abilities": ["Cleave", "Dismantle", "Malevolent Shrine", "Flame Arrow", 
                             "World Cutting Slash", "Reversed Cursed Technique"],
        "weaknesses": ["Arrogance", "Host dependency"],
        "range": ["Close", "Mid", "Long"],
        "faction": "JJK",
        "description": "The King of Curses, undisputed strongest curse in history"
    },
    "Toji Fushiguro": {
        "physical_power": 95,
        "speed": 100,  # Peak human + heavenly restriction
        "durability": 85,
        "cursed_energy": 0,  # Absolute zero - immune to cursed sensory
        "technique_versatility": 90,
        "combat_iq": 98,
        "regeneration": 10,  # Normal human
        "domain_expansion": 0,  # Cannot use domains
        "special_abilities": ["Heavenly Restriction", "Invisible to CE Sensing", 
                             "Inverted Spear of Heaven", "Playful Cloud",
                             "Cursed Spirit Manipulation Blade", "Split Soul Katana"],
        "weaknesses": ["No cursed energy defense", "Mortal durability", "No domain"],
        "range": ["Close", "Mid"],
        "faction": "JJK",
        "description": "The Sorcerer Killer, perfected physical body with zero cursed energy"
    }
}

# Demon Slayer - Upper Moon Demons
UPPER_MOONS = {
    "Muzan Kibutsuji": {
        "physical_power": 98,
        "speed": 95,
        "durability": 95,
        "cursed_energy": 0,
        "demon_power": 100,
        "technique_versatility": 95,
        "combat_iq": 90,
        "regeneration": 100,  # Near-instant regeneration
        "domain_expansion": 0,
        "special_abilities": ["Biokinesis", "Multiple Combat Forms", "Shockwave Generation", 
                             "Cell Manipulation", "Demon Creation", "Telepathy"],
        "weaknesses": ["Sunlight (CRITICAL)", "Nichirin Blades", "Wisteria Poison"],
        "range": ["Close", "Mid"],
        "faction": "DS_Demon",
        "description": "The Demon King, progenitor of all demons"
    },
    "Kokushibo (Upper Moon 1)": {
        "physical_power": 95,
        "speed": 98,
        "durability": 92,
        "cursed_energy": 0,
        "demon_power": 98,
        "technique_versatility": 95,
        "combat_iq": 95,
        "regeneration": 96,
        "domain_expansion": 0,
        "special_abilities": ["Moon Breathing - All 16 Forms", "Transparent World", 
                             "Crescent Moon Blades", "Flesh Sword Manipulation"],
        "weaknesses": ["Sunlight (CRITICAL)", "Nichirin Blades", "Decapitation"],
        "range": ["Close", "Mid", "Long"],
        "faction": "DS_Demon",
        "description": "Strongest Upper Moon, former Demon Slayer turned demon"
    },
    "Doma (Upper Moon 2)": {
        "physical_power": 88,
        "speed": 92,
        "durability": 90,
        "cursed_energy": 0,
        "demon_power": 95,
        "technique_versatility": 93,
        "combat_iq": 85,
        "regeneration": 95,
        "domain_expansion": 0,
        "special_abilities": ["Cryokinesis", "Ice Constructs", "Blood Freezing", 
                             "Crystalline Divine Child", "Frozen Lotus"],
        "weaknesses": ["Sunlight (CRITICAL)", "Nichirin Blades", "Wisteria", "Emotionless"],
        "range": ["Close", "Mid", "Long"],
        "faction": "DS_Demon",
        "description": "Upper Moon 2, master of ice Blood Demon Art"
    },
    "Akaza (Upper Moon 3)": {
        "physical_power": 96,
        "speed": 98,
        "durability": 93,
        "cursed_energy": 0,
        "demon_power": 94,
        "technique_versatility": 88,
        "combat_iq": 93,
        "regeneration": 95,
        "domain_expansion": 0,
        "special_abilities": ["Destructive Death Style", "Compass Needle Detection", 
                             "Annihilation Type", "Blue Silver Chaotic Afterglow"],
        "weaknesses": ["Sunlight (CRITICAL)", "Nichirin Blades", "Refuses to eat women"],
        "range": ["Close", "Mid"],
        "faction": "DS_Demon",
        "description": "Upper Moon 3, pure martial artist demon"
    },
    "Hantengu (Upper Moon 4)": {
        "physical_power": 70,
        "speed": 80,
        "durability": 95,  # Hidden main body
        "cursed_energy": 0,
        "demon_power": 92,
        "technique_versatility": 98,
        "combat_iq": 78,
        "regeneration": 94,
        "domain_expansion": 0,
        "special_abilities": ["Emotion Clones (Sekido, Karaku, Aizetsu, Urogi)", 
                             "Zohakuten Fusion", "Lightning", "Sound", "Wind", "Wood"],
        "weaknesses": ["Sunlight (CRITICAL)", "Nichirin Blades", "Tiny main body"],
        "range": ["Close", "Mid", "Long"],
        "faction": "DS_Demon",
        "description": "Upper Moon 4, splits into emotion-based clones"
    },
    "Gyokko (Upper Moon 5)": {
        "physical_power": 82,
        "speed": 85,
        "durability": 85,
        "cursed_energy": 0,
        "demon_power": 88,
        "technique_versatility": 90,
        "combat_iq": 72,
        "regeneration": 90,
        "domain_expansion": 0,
        "special_abilities": ["Pot Teleportation", "Water Prison Pot", 
                             "Thousand Needle Fish Kill", "True Form Transformation"],
        "weaknesses": ["Sunlight (CRITICAL)", "Nichirin Blades", "Arrogance"],
        "range": ["Close", "Mid", "Long"],
        "faction": "DS_Demon",
        "description": "Upper Moon 5, teleportation through pots"
    },
    "Gyutaro & Daki (Upper Moon 6)": {
        "physical_power": 88,
        "speed": 90,
        "durability": 92,
        "cursed_energy": 0,
        "demon_power": 90,
        "technique_versatility": 88,
        "combat_iq": 85,
        "regeneration": 93,
        "domain_expansion": 0,
        "special_abilities": ["Blood Sickles", "Flying Blood Sickles", "Poison",
                             "Obi Sash Manipulation", "Shared Life"],
        "weaknesses": ["Sunlight (CRITICAL)", "Nichirin Blades", "Must kill both simultaneously"],
        "range": ["Close", "Mid"],
        "faction": "DS_Demon",
        "description": "Upper Moon 6, sibling demons sharing one spot"
    }
}

# Demon Slayer - Hashira
HASHIRA = {
    "Gyomei Himejima": {
        "physical_power": 95,
        "speed": 88,
        "durability": 92,
        "cursed_energy": 0,
        "technique_versatility": 88,
        "combat_iq": 90,
        "regeneration": 8,
        "domain_expansion": 0,
        "special_abilities": ["Stone Breathing", "Transparent World", "Selfless State",
                             "Flail and Axe Mastery", "Mark Awakening"],
        "weaknesses": ["Human durability", "Blind", "Mortal"],
        "range": ["Close", "Mid"],
        "faction": "DS_Hashira",
        "description": "Strongest Hashira, Stone Pillar"
    },
    "Sanemi Shinazugawa": {
        "physical_power": 90,
        "speed": 92,
        "durability": 88,
        "cursed_energy": 0,
        "technique_versatility": 85,
        "combat_iq": 82,
        "regeneration": 8,
        "domain_expansion": 0,
        "special_abilities": ["Wind Breathing", "Marechi Blood (Intoxicating to demons)",
                             "Mark Awakening", "Extreme Pain Tolerance"],
        "weaknesses": ["Human durability", "Reckless fighting style"],
        "range": ["Close", "Mid"],
        "faction": "DS_Hashira",
        "description": "Wind Pillar, berserker fighting style"
    },
    "Muichiro Tokito": {
        "physical_power": 78,
        "speed": 95,
        "durability": 75,
        "cursed_energy": 0,
        "technique_versatility": 92,
        "combat_iq": 88,
        "regeneration": 6,
        "domain_expansion": 0,
        "special_abilities": ["Mist Breathing - 7th Form Creator", "Transparent World",
                             "Mark Awakening", "Prodigy"],
        "weaknesses": ["Human durability", "Young and small", "Memory issues"],
        "range": ["Close", "Mid"],
        "faction": "DS_Hashira",
        "description": "Mist Pillar, youngest prodigy"
    },
    "Obanai Iguro": {
        "physical_power": 82,
        "speed": 90,
        "durability": 80,
        "cursed_energy": 0,
        "technique_versatility": 88,
        "combat_iq": 85,
        "regeneration": 7,
        "domain_expansion": 0,
        "special_abilities": ["Serpent Breathing", "Kaburamaru (Snake companion)",
                             "Mark Awakening", "Red Nichirin Blade"],
        "weaknesses": ["Human durability", "Partial blindness"],
        "range": ["Close", "Mid"],
        "faction": "DS_Hashira",
        "description": "Serpent Pillar, technical swordsman"
    },
    "Mitsuri Kanroji": {
        "physical_power": 88,
        "speed": 90,
        "durability": 85,
        "cursed_energy": 0,
        "technique_versatility": 85,
        "combat_iq": 78,
        "regeneration": 7,
        "domain_expansion": 0,
        "special_abilities": ["Love Breathing", "Unique Muscle Composition",
                             "Whip-like Blade", "Mark Awakening"],
        "weaknesses": ["Human durability", "Naive personality"],
        "range": ["Close", "Mid"],
        "faction": "DS_Hashira",
        "description": "Love Pillar, superhuman muscle density"
    },
    "Giyu Tomioka": {
        "physical_power": 85,
        "speed": 90,
        "durability": 85,
        "cursed_energy": 0,
        "technique_versatility": 92,
        "combat_iq": 90,
        "regeneration": 7,
        "domain_expansion": 0,
        "special_abilities": ["Water Breathing - 11th Form Creator", "Dead Calm",
                             "Mark Awakening"],
        "weaknesses": ["Human durability", "Survivor's guilt"],
        "range": ["Close", "Mid"],
        "faction": "DS_Hashira",
        "description": "Water Pillar, creator of 11th Form"
    },
    "Shinobu Kocho": {
        "physical_power": 55,
        "speed": 95,
        "durability": 60,
        "cursed_energy": 0,
        "technique_versatility": 85,
        "combat_iq": 95,
        "regeneration": 5,
        "domain_expansion": 0,
        "special_abilities": ["Insect Breathing", "Wisteria Poison Expert",
                             "Stinger Thrust Style", "Pharmaceutical Knowledge"],
        "weaknesses": ["Cannot decapitate demons", "Human durability", "Small"],
        "range": ["Close"],
        "faction": "DS_Hashira",
        "description": "Insect Pillar, poison specialist"
    },
    "Tengen Uzui": {
        "physical_power": 90,
        "speed": 93,
        "durability": 88,
        "cursed_energy": 0,
        "technique_versatility": 85,
        "combat_iq": 88,
        "regeneration": 7,
        "domain_expansion": 0,
        "special_abilities": ["Sound Breathing", "Musical Score Battle Analysis",
                             "Poison Resistance", "Shinobi Training", "Dual Cleavers"],
        "weaknesses": ["Human durability", "Flashy fighting style"],
        "range": ["Close", "Mid"],
        "faction": "DS_Hashira",
        "description": "Sound Pillar, former shinobi"
    },
    "Kyojuro Rengoku": {
        "physical_power": 88,
        "speed": 90,
        "durability": 90,
        "cursed_energy": 0,
        "technique_versatility": 85,
        "combat_iq": 88,
        "regeneration": 7,
        "domain_expansion": 0,
        "special_abilities": ["Flame Breathing - All Forms Mastered", 
                             "Ninth Form: Rengoku", "Unwavering Spirit"],
        "weaknesses": ["Human durability", "Overly honorable"],
        "range": ["Close", "Mid"],
        "faction": "DS_Hashira",
        "description": "Flame Pillar, passionate warrior"
    }
}

# Combine all DS characters
DS_TEAM = {**UPPER_MOONS, **HASHIRA}

# All characters
ALL_CHARACTERS = {**JJK_TEAM, **DS_TEAM}

# ============================================================================
# MACHINE LEARNING BATTLE PREDICTOR
# ============================================================================

class JJKvsDSPredictor:
    """ML-based battle prediction system for JJK vs Demon Slayer"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'physical_power', 'speed', 'durability', 'technique_versatility',
            'combat_iq', 'regeneration', 'power_score'
        ]
        self.trained = False
        
    def get_power_score(self, char: Dict) -> float:
        """Calculate overall power score"""
        base = (char.get('physical_power', 0) * 0.15 +
                char.get('speed', 0) * 0.18 +
                char.get('durability', 0) * 0.12 +
                char.get('technique_versatility', 0) * 0.15 +
                char.get('combat_iq', 0) * 0.15 +
                char.get('regeneration', 0) * 0.10)
        
        # Special bonuses
        if char.get('domain_expansion', 0) > 0:
            base *= 1.25  # Domain expansion is massive advantage
        
        if char.get('cursed_energy', 0) > 0:
            base += char['cursed_energy'] * 0.15
        
        if char.get('demon_power', 0) > 0:
            base += char['demon_power'] * 0.12
            
        return base
    
    def create_training_data(self, n_samples: int = 2000) -> pd.DataFrame:
        """Generate synthetic training data based on character stats"""
        training_data = []
        characters = list(ALL_CHARACTERS.items())
        
        for _ in range(n_samples):
            # Pick two random characters
            idx1, idx2 = np.random.choice(len(characters), 2, replace=False)
            char1_name, char1 = characters[idx1]
            char2_name, char2 = characters[idx2]
            
            # Feature differences
            features = {
                'physical_power_diff': char1.get('physical_power', 0) - char2.get('physical_power', 0),
                'speed_diff': char1.get('speed', 0) - char2.get('speed', 0),
                'durability_diff': char1.get('durability', 0) - char2.get('durability', 0),
                'tech_diff': char1.get('technique_versatility', 0) - char2.get('technique_versatility', 0),
                'iq_diff': char1.get('combat_iq', 0) - char2.get('combat_iq', 0),
                'regen_diff': char1.get('regeneration', 0) - char2.get('regeneration', 0),
                'power_diff': self.get_power_score(char1) - self.get_power_score(char2)
            }
            
            # Calculate win probability based on stats
            score1 = self.get_power_score(char1)
            score2 = self.get_power_score(char2)
            
            # Apply matchup-specific modifiers
            mod1, mod2 = self._get_matchup_modifiers(char1, char2, char1_name, char2_name)
            score1 *= mod1
            score2 *= mod2
            
            # Winner (with some randomness for training variety)
            win_prob = 1 / (1 + np.exp(-(score1 - score2) / 15))
            winner = 1 if np.random.random() < win_prob else 0
            
            features['winner'] = winner
            training_data.append(features)
        
        return pd.DataFrame(training_data)
    
    def _get_matchup_modifiers(self, char1: Dict, char2: Dict, 
                               name1: str, name2: str) -> Tuple[float, float]:
        """Calculate matchup-specific modifiers"""
        mod1, mod2 = 1.0, 1.0
        
        # JJK vs Demon specific interactions
        is_jjk_1 = char1.get('faction') == 'JJK'
        is_demon_2 = char2.get('faction') == 'DS_Demon'
        is_hashira_2 = char2.get('faction') == 'DS_Hashira'
        
        is_jjk_2 = char2.get('faction') == 'JJK'
        is_demon_1 = char1.get('faction') == 'DS_Demon'
        is_hashira_1 = char1.get('faction') == 'DS_Hashira'
        
        # Domain Expansion is extremely powerful
        if char1.get('domain_expansion', 0) > 80:
            mod1 *= 1.35  # Domains can trap and guarantee hits
            
        if char2.get('domain_expansion', 0) > 80:
            mod2 *= 1.35
        
        # Cursed energy vs demons
        if is_jjk_1 and is_demon_2 and char1.get('cursed_energy', 0) > 50:
            mod1 *= 1.15  # Cursed energy effective against demons
        
        if is_jjk_2 and is_demon_1 and char2.get('cursed_energy', 0) > 50:
            mod2 *= 1.15
        
        # Sukuna specific bonuses
        if 'Sukuna' in name1:
            mod1 *= 1.15  # King of Curses factor
            if is_demon_2:
                mod1 *= 1.10  # Cleave/Dismantle destroys regeneration
        
        if 'Sukuna' in name2:
            mod2 *= 1.15
            if is_demon_1:
                mod2 *= 1.10
        
        # Toji specific matchups
        if 'Toji' in name1:
            mod1 *= 1.08  # Heavenly restriction combat boost
            if is_demon_2:
                mod1 *= 1.05  # Invisible to sensing
                mod1 *= 0.90  # But can't permanently kill demons
            if is_hashira_2:
                mod1 *= 1.15  # Heavily outclasses humans
        
        if 'Toji' in name2:
            mod2 *= 1.08
            if is_demon_1:
                mod2 *= 1.05
                mod2 *= 0.90
            if is_hashira_1:
                mod2 *= 1.15
        
        # Demon regeneration factor
        if is_demon_1 and char1.get('regeneration', 0) > 90:
            mod1 *= 1.12
        if is_demon_2 and char2.get('regeneration', 0) > 90:
            mod2 *= 1.12
        
        # Hashira marks and abilities
        if is_hashira_1:
            mod1 *= 0.95  # Human limitations
        if is_hashira_2:
            mod2 *= 0.95
            
        return mod1, mod2
    
    def train(self, n_samples: int = 3000):
        """Train the ML model"""
        print("=" * 70)
        print("TRAINING MACHINE LEARNING BATTLE PREDICTOR")
        print("=" * 70)
        
        # Generate training data
        print("\nGenerating training data...")
        df = self.create_training_data(n_samples)
        
        feature_cols = [col for col in df.columns if col != 'winner']
        X = df[feature_cols]
        y = df['winner']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("Training Random Forest Classifier...")
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_acc = self.model.score(X_train_scaled, y_train)
        test_acc = self.model.score(X_test_scaled, y_test)
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        print(f"\nModel Training Complete!")
        print(f"Training Accuracy: {train_acc:.3f}")
        print(f"Test Accuracy: {test_acc:.3f}")
        print(f"Cross-Validation: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop Feature Importances:")
        for _, row in importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        self.trained = True
        print("\n" + "=" * 70)
    
    def predict_battle(self, char1: Dict, char2: Dict, 
                      name1: str, name2: str) -> Dict:
        """Predict battle outcome between two characters"""
        if not self.trained:
            self.train()
        
        # Calculate features
        features = np.array([[
            char1.get('physical_power', 0) - char2.get('physical_power', 0),
            char1.get('speed', 0) - char2.get('speed', 0),
            char1.get('durability', 0) - char2.get('durability', 0),
            char1.get('technique_versatility', 0) - char2.get('technique_versatility', 0),
            char1.get('combat_iq', 0) - char2.get('combat_iq', 0),
            char1.get('regeneration', 0) - char2.get('regeneration', 0),
            self.get_power_score(char1) - self.get_power_score(char2)
        ]])
        
        # Scale and predict
        features_scaled = self.scaler.transform(features)
        prob = self.model.predict_proba(features_scaled)[0]
        
        # Apply matchup modifiers to final probability
        mod1, mod2 = self._get_matchup_modifiers(char1, char2, name1, name2)
        
        # Adjust probabilities based on modifiers
        base_prob1 = prob[1]
        adjusted_prob1 = base_prob1 * mod1 / (base_prob1 * mod1 + (1 - base_prob1) * mod2)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(char1, char2, name1, name2)
        
        winner = name1 if adjusted_prob1 > 0.5 else name2
        confidence = max(adjusted_prob1, 1 - adjusted_prob1)
        
        return {
            'char1': name1,
            'char2': name2,
            'char1_win_prob': round(adjusted_prob1 * 100, 1),
            'char2_win_prob': round((1 - adjusted_prob1) * 100, 1),
            'winner': winner,
            'confidence': 'High' if confidence > 0.7 else 'Moderate' if confidence > 0.55 else 'Low',
            'reasoning': reasoning,
            'char1_power': round(self.get_power_score(char1), 1),
            'char2_power': round(self.get_power_score(char2), 1)
        }
    
    def _generate_reasoning(self, char1: Dict, char2: Dict, 
                           name1: str, name2: str) -> List[str]:
        """Generate battle reasoning"""
        reasons = []
        
        # Speed comparison
        speed_diff = char1.get('speed', 0) - char2.get('speed', 0)
        if abs(speed_diff) > 10:
            faster = name1 if speed_diff > 0 else name2
            reasons.append(f"{faster} has significant speed advantage")
        
        # Power comparison
        power_diff = char1.get('physical_power', 0) - char2.get('physical_power', 0)
        if abs(power_diff) > 10:
            stronger = name1 if power_diff > 0 else name2
            reasons.append(f"{stronger} has superior physical power")
        
        # Regeneration
        regen_diff = char1.get('regeneration', 0) - char2.get('regeneration', 0)
        if abs(regen_diff) > 30:
            if regen_diff > 0:
                reasons.append(f"{name1}'s regeneration provides massive endurance advantage")
            else:
                reasons.append(f"{name2}'s regeneration provides massive endurance advantage")
        
        # Domain expansion
        if char1.get('domain_expansion', 0) > 80:
            reasons.append(f"{name1}'s Domain Expansion creates overwhelming advantage")
        if char2.get('domain_expansion', 0) > 80:
            reasons.append(f"{name2}'s Domain Expansion creates overwhelming advantage")
        
        # Special matchup notes
        if 'Sukuna' in name1 and char2.get('faction') == 'DS_Demon':
            reasons.append(f"Sukuna's Cleave can bypass demon regeneration")
        if 'Toji' in name1:
            if char2.get('faction') == 'DS_Demon':
                reasons.append(f"Toji is invisible to demon sensing but lacks means to kill permanently")
            if char2.get('faction') == 'DS_Hashira':
                reasons.append(f"Toji's Heavenly Restriction gives him superhuman advantage over regular humans")
        
        return reasons if reasons else ["Close matchup based on overall stats"]


# ============================================================================
# WAR SIMULATOR
# ============================================================================

class FullWarSimulator:
    """Simulates a full-scale war between JJK and Demon Slayer"""
    
    def __init__(self, predictor: JJKvsDSPredictor):
        self.predictor = predictor
        
    def simulate_war(self) -> Dict:
        """Simulate the full war"""
        print("\n" + "=" * 70)
        print("FULL WAR SIMULATION: JJK (Sukuna & Toji) vs DEMON SLAYER")
        print("=" * 70)
        
        results = {
            'individual_matchups': [],
            'phases': [],
            'final_outcome': None
        }
        
        # Phase 1: Individual matchup predictions
        print("\n" + "-" * 70)
        print("PHASE 1: KEY INDIVIDUAL MATCHUP PREDICTIONS")
        print("-" * 70)
        
        jjk_names = list(JJK_TEAM.keys())
        upper_moon_names = list(UPPER_MOONS.keys())
        hashira_names = list(HASHIRA.keys())
        
        # Sukuna vs all Upper Moons
        for um_name in upper_moon_names:
            result = self.predictor.predict_battle(
                JJK_TEAM["Sukuna (20 Fingers)"], 
                UPPER_MOONS[um_name],
                "Sukuna (20 Fingers)", 
                um_name
            )
            results['individual_matchups'].append(result)
            self._print_matchup(result)
        
        # Sukuna vs top Hashira
        for h_name in ["Gyomei Himejima", "Sanemi Shinazugawa", "Muichiro Tokito"]:
            result = self.predictor.predict_battle(
                JJK_TEAM["Sukuna (20 Fingers)"],
                HASHIRA[h_name],
                "Sukuna (20 Fingers)",
                h_name
            )
            results['individual_matchups'].append(result)
            self._print_matchup(result)
        
        # Toji vs Upper Moons
        for um_name in upper_moon_names:
            result = self.predictor.predict_battle(
                JJK_TEAM["Toji Fushiguro"],
                UPPER_MOONS[um_name],
                "Toji Fushiguro",
                um_name
            )
            results['individual_matchups'].append(result)
            self._print_matchup(result)
        
        # Toji vs all Hashira
        for h_name in hashira_names:
            result = self.predictor.predict_battle(
                JJK_TEAM["Toji Fushiguro"],
                HASHIRA[h_name],
                "Toji Fushiguro",
                h_name
            )
            results['individual_matchups'].append(result)
            self._print_matchup(result)
        
        # Phase 2: Team battles
        print("\n" + "-" * 70)
        print("PHASE 2: TEAM BATTLE SIMULATIONS")
        print("-" * 70)
        
        # Calculate total team powers
        jjk_power = sum(self.predictor.get_power_score(c) for c in JJK_TEAM.values())
        um_power = sum(self.predictor.get_power_score(c) for c in UPPER_MOONS.values())
        hashira_power = sum(self.predictor.get_power_score(c) for c in HASHIRA.values())
        ds_total_power = um_power + hashira_power
        
        # Apply faction modifiers
        jjk_power *= 1.30  # Domain expansion advantage
        um_power *= 1.15   # Regeneration advantage
        hashira_power *= 0.92  # Human limitations
        
        print(f"\nTeam Power Levels (with modifiers):")
        print(f"  JJK Team (Sukuna + Toji): {jjk_power:.1f}")
        print(f"  Upper Moons Total: {um_power:.1f}")
        print(f"  Hashira Total: {hashira_power:.1f}")
        print(f"  DS Combined: {um_power + hashira_power:.1f}")
        
        # Phase 3: War scenarios
        results['phases'] = self._simulate_war_phases(jjk_power, um_power, hashira_power)
        
        # Phase 4: Final outcome
        print("\n" + "-" * 70)
        print("PHASE 3: FINAL WAR OUTCOME")
        print("-" * 70)
        
        results['final_outcome'] = self._determine_final_outcome(results)
        self._print_final_outcome(results['final_outcome'])
        
        return results
    
    def _print_matchup(self, result: Dict):
        """Print a single matchup result"""
        winner_char = ">>>" if result['winner'] == result['char1'] else "<<<"
        print(f"\n{result['char1']} vs {result['char2']}")
        print(f"  Winner: {result['winner']} ({result['confidence']} confidence)")
        print(f"  Win Probabilities: {result['char1']}: {result['char1_win_prob']}% | {result['char2']}: {result['char2_win_prob']}%")
        print(f"  Power Scores: {result['char1_power']} vs {result['char2_power']}")
        for reason in result['reasoning'][:2]:  # Show top 2 reasons
            print(f"    - {reason}")
    
    def _simulate_war_phases(self, jjk_power: float, um_power: float, 
                             hashira_power: float) -> List[Dict]:
        """Simulate war in phases"""
        phases = []
        
        # Phase 1: JJK vs Hashira
        print(f"\n[Scenario 1] Sukuna & Toji vs All Hashira:")
        ratio1 = jjk_power / hashira_power
        winner1 = "JJK" if ratio1 > 1 else "Hashira"
        margin1 = "Decisive" if ratio1 > 1.5 or ratio1 < 0.67 else "Moderate"
        print(f"  Power Ratio: {ratio1:.2f}")
        print(f"  Outcome: {winner1} Victory ({margin1})")
        phases.append({'name': 'vs Hashira', 'winner': winner1, 'ratio': ratio1})
        
        # Phase 2: JJK vs Upper Moons
        print(f"\n[Scenario 2] Sukuna & Toji vs All Upper Moons:")
        ratio2 = jjk_power / um_power
        winner2 = "JJK" if ratio2 > 1 else "Upper Moons"
        margin2 = "Decisive" if ratio2 > 1.5 or ratio2 < 0.67 else "Moderate"
        print(f"  Power Ratio: {ratio2:.2f}")
        print(f"  Outcome: {winner2} Victory ({margin2})")
        phases.append({'name': 'vs Upper Moons', 'winner': winner2, 'ratio': ratio2})
        
        # Phase 3: JJK vs All Demon Slayer
        print(f"\n[Scenario 3] Sukuna & Toji vs All Upper Moons + Hashira:")
        combined_power = um_power + hashira_power * 0.7  # Hashira less effective without demon coordination
        ratio3 = jjk_power / combined_power
        winner3 = "JJK" if ratio3 > 0.85 else "Demon Slayer"
        margin3 = "Decisive" if ratio3 > 1.3 or ratio3 < 0.5 else "Hard-fought"
        print(f"  Power Ratio: {ratio3:.2f}")
        print(f"  Outcome: {winner3} Victory ({margin3})")
        phases.append({'name': 'vs All DS', 'winner': winner3, 'ratio': ratio3})
        
        return phases
    
    def _determine_final_outcome(self, results: Dict) -> Dict:
        """Determine the final war outcome"""
        # Count wins from individual matchups
        sukuna_wins = sum(1 for m in results['individual_matchups'] 
                        if m['winner'] == 'Sukuna (20 Fingers)')
        toji_wins = sum(1 for m in results['individual_matchups'] 
                       if m['winner'] == 'Toji Fushiguro')
        jjk_total_wins = sukuna_wins + toji_wins
        
        total_matchups = len(results['individual_matchups'])
        ds_wins = total_matchups - jjk_total_wins
        
        # Phase wins
        jjk_phase_wins = sum(1 for p in results['phases'] if 'JJK' in p['winner'])
        
        # Determine overall victor
        if jjk_total_wins > total_matchups * 0.6 and jjk_phase_wins >= 2:
            overall_winner = "JJK (Sukuna & Toji)"
            victory_type = "Decisive Victory"
        elif jjk_total_wins > total_matchups * 0.5:
            overall_winner = "JJK (Sukuna & Toji)"
            victory_type = "Close Victory"
        elif ds_wins > jjk_total_wins:
            overall_winner = "Demon Slayer Coalition"
            victory_type = "War of Attrition"
        else:
            overall_winner = "JJK (Sukuna & Toji)"
            victory_type = "Pyrrhic Victory"
        
        return {
            'winner': overall_winner,
            'victory_type': victory_type,
            'jjk_matchup_wins': jjk_total_wins,
            'ds_matchup_wins': ds_wins,
            'sukuna_record': f"{sukuna_wins}-{total_matchups - sukuna_wins - toji_wins + ds_wins - (total_matchups - jjk_total_wins - (total_matchups - jjk_total_wins))}",
            'toji_record': f"{toji_wins}-{len([m for m in results['individual_matchups'] if 'Toji' in m['char1']]) - toji_wins}"
        }
    
    def _print_final_outcome(self, outcome: Dict):
        """Print the final war outcome"""
        print(f"\n{'='*70}")
        print(f"FINAL WAR OUTCOME: {outcome['winner']}")
        print(f"{'='*70}")
        print(f"\nVictory Type: {outcome['victory_type']}")
        print(f"Total Matchup Score: JJK {outcome['jjk_matchup_wins']} - {outcome['ds_matchup_wins']} Demon Slayer")
        
        print(f"\n" + "="*70)
        print("WAR NARRATIVE")
        print("="*70)
        
        narrative = """
The great cross-dimensional war between the forces of Jujutsu Kaisen and 
Demon Slayer has concluded after an epic series of battles.

SUKUNA'S CAMPAIGN:
The King of Curses proved to be an overwhelming force. His Domain Expansion,
Malevolent Shrine, created inescapable killing fields where even the Upper 
Moons' legendary regeneration faltered against Cleave and Dismantle. Muzan
Kibutsuji himself could not match the raw cursed energy output, and the 
Hashira were simply outclassed by divine-level power.

TOJI'S EFFECTIVENESS:
The Sorcerer Killer lived up to his name, decimating the Hashira with his
Heavenly Restriction-enhanced physicality. Against demons, his speed made
him nearly impossible to track, though his lack of cursed energy meant he
could wound but not permanently destroy Upper Moon-level threats.

KEY TURNING POINTS:
1. Sukuna's Domain Expansion overwhelmed all demon Blood Demon Arts
2. Toji's Split Soul Katana proved effective even against demon souls
3. The Upper Moons' regeneration couldn't keep pace with Sukuna's slashing
4. Hashira marks and Transparent World were insufficient against JJK-level threats

CONCLUSION:
While the Demon Slayer forces showed incredible resilience, the power scaling
difference between cursed energy users and demon physiology proved decisive.
Sukuna alone could likely defeat the entire Demon Slayer roster, while Toji
served as an extremely effective force multiplier against human-level threats.
"""
        print(narrative)
        

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("JJK vs DEMON SLAYER: ULTIMATE WAR SIMULATION")
    print("Sukuna & Toji vs Upper Moon Demons & Hashira")
    print("="*70)
    
    # Initialize and train predictor
    predictor = JJKvsDSPredictor()
    predictor.train(n_samples=3000)
    
    # Run war simulation
    simulator = FullWarSimulator(predictor)
    results = simulator.simulate_war()
    
    # Print summary statistics
    print("\n" + "="*70)
    print("BATTLE STATISTICS SUMMARY")
    print("="*70)
    
    sukuna_matchups = [m for m in results['individual_matchups'] if m['char1'] == 'Sukuna (20 Fingers)']
    toji_matchups = [m for m in results['individual_matchups'] if m['char1'] == 'Toji Fushiguro']
    
    print(f"\nSUKUNA'S RECORD:")
    sukuna_wins = sum(1 for m in sukuna_matchups if m['winner'] == 'Sukuna (20 Fingers)')
    print(f"  Wins: {sukuna_wins} / {len(sukuna_matchups)}")
    print(f"  Average Win Probability: {np.mean([m['char1_win_prob'] for m in sukuna_matchups]):.1f}%")
    
    print(f"\nTOJI'S RECORD:")
    toji_wins = sum(1 for m in toji_matchups if m['winner'] == 'Toji Fushiguro')
    print(f"  Wins: {toji_wins} / {len(toji_matchups)}")
    print(f"  Average Win Probability: {np.mean([m['char1_win_prob'] for m in toji_matchups]):.1f}%")
    
    # Toji vs Hashira breakdown
    toji_vs_hashira = [m for m in toji_matchups if m['char2'] in HASHIRA]
    toji_vs_um = [m for m in toji_matchups if m['char2'] in UPPER_MOONS]
    
    print(f"\nTOJI vs HASHIRA: {sum(1 for m in toji_vs_hashira if m['winner'] == 'Toji Fushiguro')}/{len(toji_vs_hashira)}")
    print(f"TOJI vs UPPER MOONS: {sum(1 for m in toji_vs_um if m['winner'] == 'Toji Fushiguro')}/{len(toji_vs_um)}")
    
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    
    return results


if __name__ == "__main__":
    results = main()
