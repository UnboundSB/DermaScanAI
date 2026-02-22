import random

class SkincareRecommender:
    def __init__(self):
        # --- 1. INTROS ---
        self.intros = [
            "Ah! ", "Oh my, ", "I can see that you have ", 
            "Looking closely, it appears you are dealing with ",
            "Based on the scan, I noticed ", "It looks like you might be experiencing ",
            "My analysis shows signs of ", "I am detecting ",
            "Upon reviewing your skin, I see ", "It is quite clear that there is ",
            "I've carefully analyzed your face and found ", "Well, it seems we have ",
            "Let's talk about the ", "I can spot ", "The results are in, and I'm seeing "
        ]
        
        # --- 2. SEVERITY ADJECTIVES ---
        self.severity_adjectives = {
            "minor": [
                "a little bit of ", "minor ", "slight ", "early-stage ", "a very mild case of ", 
                "faint ", "surface-level ", "barely noticeable ", "a touch of ", "very light ", 
                "mild ", "superficial ", "a hint of ", "beginning signs of ", "low-level "
            ],
            "moderate": [
                "moderate ", "noticeable ", "developing ", "visible ", "a fair amount of ", 
                "distinct ", "active ", "mid-level ", "established ", "clear signs of ", 
                "average ", "standard ", "unmistakable ", "persistent ", "ongoing "
            ],
            "severe": [
                "severe ", "highly visible ", "significant ", "prominent ", "advanced ", 
                "heavy ", "intense ", "deep-rooted ", "stubborn ", "widespread ", 
                "pronounced ", "major ", "aggressive ", "acute ", "strong "
            ]
        }
        
        # --- 3. TRANSITIONS ---
        self.transitions = [
            " I highly recommend using ", " I suggest applying ", 
            " You should definitely look into ", " The best approach would be to use ",
            " A great step forward would be incorporating ", " You might want to try ",
            " Consider adding ", " My top advice is to start with ",
            " Let's focus on treating this with ", " Your skin would really benefit from ",
            " It would be wise to utilize ", " We can manage this effectively by applying ",
            " A solid routine for this includes ", " You'll see great improvements by using ",
            " The gold standard here is "
        ]
        
        # --- 4. OUTROS ---
        self.outros = [
            " to overcome this.", " to treat the area.", " to rejuvenate your skin.", 
            " to clear this up effectively.", " for the best possible results.",
            " to restore your natural glow.", " to see a noticeable improvement.",
            " to help balance your complexion.", " and bring your skin back to life.",
            " so you can feel confident again.", " for long-lasting skin health.",
            " to gently soothe the affected areas.", " to effectively combat these symptoms.",
            " and keep your skin barrier strong.", " to achieve a smoother texture."
        ]

        # --- 5. TREATMENTS ---
        self.treatments = {
            "acne": {
                "minor": [
                    "a gentle 2% Salicylic Acid cleanser", "a mild tea tree oil spot treatment", 
                    "a niacinamide balancing serum", "a witch hazel clarifying toner", "a gentle BHA liquid exfoliant"
                ],
                "moderate": [
                    "a targeted 5% Benzoyl Peroxide gel", "an Azelaic Acid 10% suspension", 
                    "a sulfur-based clarifying mask", "an Adapalene 0.1% gel", "a PHA/BHA exfoliating toner"
                ],
                "severe": [
                    "a prescription-strength Tretinoin cream", "an oral antibiotic combined with topical Clindamycin", 
                    "a high-strength 10% Benzoyl Peroxide wash", "a potent clinical Retinoid", "an Isotretinoin consultation with a dermatologist"
                ]
            },
            "darkspots": {
                "minor": [
                    "a daily 10% Vitamin C serum", "a licorice root extract essence", 
                    "a mild AHA exfoliating toner", "a brightening Alpha Arbutin drop", "a soothing turmeric mask"
                ],
                "moderate": [
                    "a serum combining Niacinamide and Alpha Arbutin", "a Kojic Acid cleansing bar", 
                    "a 15% Vitamin C serum with Ferulic Acid", "a Glycolic Acid overnight peel", "a 10% Niacinamide booster"
                ],
                "severe": [
                    "a potent Tranexamic Acid treatment with strict SPF 50", "a Hydroquinone 2% cream", 
                    "an intense pulsed light (IPL) therapy consultation", "a high-strength chemical peel (like TCA)", "a prescription cysteamine cream"
                ]
            },
            "puffy_eyes": {
                "minor": [
                    "a cooling caffeine eye roller", "a cucumber extract soothing gel", 
                    "chilled under-eye hydrogel patches", "a green tea infused eye cream", "a simple cold compress routine"
                ],
                "moderate": [
                    "a peptide-rich eye cream", "an Arnica infused eye serum", 
                    "lymphatic drainage massage tools like Gua Sha", "a plumping Hyaluronic Acid eye mask", "a Vitamin K brightening eye cream"
                ],
                "severe": [
                    "a targeted Hyaluronic Acid eye serum", "a concentrated Retinol eye serum", 
                    "a targeted micro-current device treatment", "a specialized tear-trough filler consultation", "a high-grade medical eye complex with Haloxyl"
                ]
            },
            "wrinkles": {
                "minor": [
                    "a hydrating Hyaluronic Acid serum", "a ceramide-rich daily moisturizer", 
                    "an antioxidant CoQ10 serum", "a gentle Bakuchiol oil", "a daily SPF 50 with added peptides"
                ],
                "moderate": [
                    "an over-the-counter Retinol 0.5% cream", "an Argireline peptide solution", 
                    "a Glycolic Acid resurfacing pad routine", "a Copper Peptide serum", "an encapsulated Retinaldehyde cream"
                ],
                "severe": [
                    "a prescription-strength Tretinoin 0.05% cream", "professional microneedling sessions", 
                    "a fractional laser treatment consultation", "a neuromodulator (Botox) consultation", "an intense collagen-stimulating radiofrequency treatment"
                ]
            }
        }
        
        self.clear_face_maintenance = [
            "a gentle hydrating cleanser", "a daily SPF 50 sunscreen", "a lightweight ceramide moisturizer", 
            "a weekly antioxidant mask", "a simple barrier-repairing serum", "a Vitamin C morning serum", 
            "a hydrating mist throughout the day", "a silk pillowcase to prevent friction", "a double-cleansing routine at night", 
            "a barrier-protecting Squalane oil", "a green tea soothing essence", "a deeply nourishing overnight mask", 
            "a balancing rosewater toner", "a gentle enzyme exfoliant once a week", "a fragrance-free moisturizer"
        ]

    def _determine_severity(self, confidence: float) -> str:
        """Converts the AI's percentage into a severity bracket."""
        if confidence < 60.0:
            return "minor"
        elif confidence < 85.0:
            return "moderate"
        else:
            return "severe"

    def _format_multiple_treatments(self, selected_treatments: list) -> str:
        """Properly joins multiple treatments with commas, 'and', or 'or'."""
        if len(selected_treatments) == 1:
            return selected_treatments[0]
        elif len(selected_treatments) == 2:
            conjunction = random.choice([" and ", " or "])
            return conjunction.join(selected_treatments)
        else:
            # For 3 items: "A, B, and C"
            return ", ".join(selected_treatments[:-1]) + ", and " + selected_treatments[-1]

    def generate_prescription(self, symptom: str, confidence: float) -> str:
        """Assembles the randomized NLP sentence with multiple treatment options."""
        formatted_symptom = symptom.replace("_", " ")
        intro = random.choice(self.intros)
        transition = random.choice(self.transitions)
        outro = random.choice(self.outros)
        
        # Decide whether to suggest 2 or 3 treatments to make it sound like a full regimen
        num_suggestions = random.choice([2, 3])
        
        if symptom == "clear_face":
            selected_maintenance = random.sample(self.clear_face_maintenance, num_suggestions)
            treatment_text = self._format_multiple_treatments(selected_maintenance)
            return f"Ah! Your skin looks incredibly healthy and clear. To maintain this, {transition.strip().lower()} {treatment_text}."

        severity_level = self._determine_severity(confidence)
        adj = random.choice(self.severity_adjectives[severity_level])
        
        # Randomly grab 2 or 3 distinct treatments from the severity bracket
        selected_treatments = random.sample(self.treatments[symptom][severity_level], num_suggestions)
        treatment_text = self._format_multiple_treatments(selected_treatments)
        
        final_sentence = f"{intro}{adj}{formatted_symptom}.{transition}{treatment_text}{outro}"
        
        return final_sentence.capitalize()

# --- QUICK TEST HARNESS ---
if __name__ == "__main__":
    doc = SkincareRecommender()
    
    test_cases = [
        ("acne", 92.5),       
        ("darkspots", 45.0),  
        ("puffy_eyes", 75.0), 
        ("wrinkles", 88.0),   
        ("clear_face", 98.0)  
    ]
    
    print("--- MULTI-OPTION NLP DOCTOR TEST ---")
    for symp, conf in test_cases:
        sentence = doc.generate_prescription(symp, conf)
        print(f"\n[Input: {symp} @ {conf}%]")
        print(f"Output: {sentence}")