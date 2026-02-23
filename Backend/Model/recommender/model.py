import random

class SkincareRecommender:
    def __init__(self):
        # --- THE KNOWLEDGE BASE ---
        self.intros = [
            "Ah! ", "Oh my, ", "I can see that you have ", 
            "Looking closely, it appears you are dealing with ",
            "Based on the scan, I noticed ", "It looks like you might be experiencing ",
            "My analysis shows signs of ", "I am detecting ",
            "Upon reviewing your skin, I see ", "It is quite clear that there is ",
            "I've carefully analyzed your face and found ", "Well, it seems we have ",
            "Right away, the scan reveals ", "It's evident that we are looking at ",
            "The data points toward ", "I've picked up on some ",
            "The imaging confirms the presence of ", "I am noticing distinct areas of "
        ]
        
        self.severity_adjectives = {
            "minor": [
                "a little bit of ", "minor ", "slight ", "early-stage ", 
                "a very mild case of ", "surface-level ", "faint traces of ",
                "barely noticeable ", "a touch of ", "beginning signs of "
            ],
            "moderate": [
                "moderate ", "noticeable ", "developing ", "visible ", 
                "a fair amount of ", "persistent ", "stubborn ",
                "distinct ", "active ", "mid-level "
            ],
            "severe": [
                "severe ", "highly visible ", "significant ", "prominent ", 
                "advanced ", "deeply rooted ", "intense ",
                "widespread ", "pronounced ", "aggressive "
            ]
        }
        
        self.transitions = [
            " I highly recommend using ", " I suggest applying ", 
            " You should definitely look into ", " The best approach would be to use ",
            " A great step forward would be incorporating ", " You might want to try ",
            " Let's get that under control with ", " A solid game plan involves ",
            " The most effective route here is ", " Your skin will really benefit from ",
            " We can tackle this directly by introducing ", " My top clinical recommendation is "
        ]

        self.outros = [
            " to overcome this.", " to treat the area.", " to rejuvenate your skin.", 
            " to clear this up effectively.", " for the best possible results.",
            " to restore your natural balance.", " to see a noticeable improvement.",
            " to help balance your complexion.", " and bring your skin back to life.",
            " to stop this at the source.", " for optimal skin health."
        ]

        self.treatments = {
            "acne": {
                "minor": [
                    "a gentle 2% Salicylic Acid cleanser", "a mild tea tree oil spot treatment", 
                    "a niacinamide balancing serum", "a witch hazel clarifying toner", 
                    "a gentle BHA liquid exfoliant", "a hypochlorous acid rescue spray", 
                    "hydrocolloid pimple patches", "a 10% Azelaic Acid balancing serum", 
                    "a Zinc PCA daily toner", "a succinic acid blemish treatment", 
                    "a gentle sulfur-based soap bar", "a centella asiatica calming ampoule"
                ],
                "moderate": [
                    "a targeted 5% Benzoyl Peroxide gel", "an Azelaic Acid 10% suspension", 
                    "a sulfur-based clarifying mask", "an Adapalene 0.1% gel", 
                    "a PHA/BHA exfoliating toner", "an encapsulated Retinol resurfacing serum", 
                    "a 2% Salicylic Acid wash left on for two minutes", "a Mandelic Acid gentle chemical peel", 
                    "a 15% Niacinamide clearing serum", "a short-contact 4% Benzoyl Peroxide cream"
                ],
                "severe": [
                    "a prescription-strength Tretinoin cream", "an oral antibiotic combined with topical Clindamycin", 
                    "a high-strength 10% Benzoyl Peroxide wash", "a potent clinical Retinoid", 
                    "an Isotretinoin consultation", "a Spironolactone consultation for hormonal breakouts", 
                    "a topical Clascoterone (Winlevi) cream", "a course of professional photodynamic therapy (PDT)", 
                    "a high-grade clinical Salicylic Acid peel", "an intralesional cortisone injection consultation"
                ]
            },
            "darkspots": {
                "minor": [
                    "a daily 10% Vitamin C serum", "a licorice root extract essence", 
                    "a mild AHA exfoliating toner", "a brightening Alpha Arbutin drop", 
                    "a soothing turmeric mask", "a 2% Alpha Arbutin brightening serum", 
                    "a daily tinted mineral sunscreen to block visible light", "a 5% Lactic Acid hydrating serum", 
                    "a gentle Kojic Acid cleansing bar", "a Kakadu Plum antioxidant essence"
                ],
                "moderate": [
                    "a serum combining Niacinamide and Alpha Arbutin", "a 15% Vitamin C serum with Ferulic Acid", 
                    "a Glycolic Acid overnight peel", "a 10% Niacinamide booster", 
                    "a 5% Cysteamine leave-on cream", "a 3% Tranexamic Acid targeted treatment", 
                    "a Hexylresorcinol dark spot corrector", "an 8% Ascorbic Acid + Alpha Arbutin formulation", 
                    "a 15% Azelaic Acid prescription gel", "a multi-acid brightening night cream"
                ],
                "severe": [
                    "a potent Tranexamic Acid treatment", "a Hydroquinone 2% cream", 
                    "an intense pulsed light (IPL) therapy consultation", "a high-strength chemical peel", 
                    "a prescription cysteamine cream", "a prescription Tri-Luma cream cycle", 
                    "a professional Q-switched Nd:YAG laser session", "medical microneedling with depigmenting serums", 
                    "a 4% prescription Hydroquinone cycle", "a professional Cosmelan depigmentation peel"
                ]
            },
            "puffy_eyes": {
                "minor": [
                    "a cooling caffeine eye roller", "a cucumber extract soothing gel", 
                    "chilled under-eye hydrogel patches", "a green tea infused eye cream", 
                    "a simple cold compress routine", "a refrigerated jade roller massage", 
                    "a cooling metal-tip eye gel applicator", "an Aloe Vera soothing eye gel", 
                    "a frozen spoon morning massage technique"
                ],
                "moderate": [
                    "a peptide-rich eye cream", "an Arnica infused eye serum", 
                    "lymphatic drainage massage tools like Gua Sha", "a plumping Hyaluronic Acid eye mask", 
                    "a Vitamin K brightening eye cream", "a 5% Caffeine + EGCG eye serum", 
                    "a Ginseng root extract firming cream", "a tightening multi-peptide eye complex", 
                    "a niacinamide-based eye brightener"
                ],
                "severe": [
                    "a targeted Hyaluronic Acid eye serum", "a concentrated Retinol eye serum", 
                    "a targeted micro-current device treatment", "a specialized tear-trough filler consultation", 
                    "a high-grade medical eye complex", "a professional radiofrequency skin tightening session", 
                    "a lower blepharoplasty consultation", "professional facial lymphatic drainage massages", 
                    "a fractional CO2 laser session specifically for the peri-orbital area"
                ]
            },
            "wrinkles": {
                "minor": [
                    "a hydrating Hyaluronic Acid serum", "a ceramide-rich daily moisturizer", 
                    "an antioxidant CoQ10 serum", "a gentle Bakuchiol oil", 
                    "a daily SPF 50 with added peptides", "a Matrixyl 3000 peptide serum", 
                    "a daily antioxidant defense mist", "a multi-weight Hyaluronic Acid essence", 
                    "a snail mucin power essence", "a pure Squalane oil for barrier defense"
                ],
                "moderate": [
                    "an over-the-counter Retinol 0.5% cream", "an Argireline peptide solution", 
                    "a Glycolic Acid resurfacing pad routine", "a Copper Peptide serum", 
                    "an encapsulated Retinaldehyde cream", "a 0.1% Retinaldehyde night serum", 
                    "an Epidermal Growth Factor (EGF) cellular serum", "an at-home FDA-cleared LED red light mask", 
                    "a 10% Lactic Acid overnight serum", "a firming vegan collagen cream"
                ],
                "severe": [
                    "a prescription-strength Tretinoin 0.05% cream", "professional microneedling sessions", 
                    "a fractional laser treatment consultation", "a neuromodulator (Botox) consultation", 
                    "an intense collagen-stimulating radiofrequency treatment", "a fractional CO2 laser resurfacing treatment", 
                    "a deep TCA (Trichloroacetic acid) professional peel", "a PDO thread lift consultation", 
                    "a prescription Tazarotene 0.1% cream", "an ultrasound skin tightening (Ultherapy) consultation"
                ]
            }
        }
        
        self.clear_face_maintenance = [
            "a gentle hydrating cleanser", "a daily SPF 50 sunscreen", 
            "a lightweight ceramide moisturizer", "a weekly antioxidant mask", 
            "a simple barrier-repairing serum", "a Vitamin C morning serum",
            "a gentle PHA daily toner", "a barrier-supporting squalane oil",
            "a silk pillowcase to prevent friction", "a double-cleansing routine at night"
        ]

    def _determine_severity(self, confidence: float) -> str:
        if confidence < 60.0: return "minor"
        elif confidence < 85.0: return "moderate"
        else: return "severe"

    def _format_multiple_treatments(self, selected_treatments: list) -> str:
        if len(selected_treatments) == 1:
            return selected_treatments[0]
        elif len(selected_treatments) == 2:
            conjunction = random.choice([" and ", " or "])
            return conjunction.join(selected_treatments)
        else:
            return ", ".join(selected_treatments[:-1]) + ", and " + selected_treatments[-1]

    def generate_prescription(self, diagnoses: list) -> str:
        intro = random.choice(self.intros)
        transition = random.choice(self.transitions)
        outro = random.choice(self.outros)
        
        if diagnoses[0][0] == "clear_face":
            num_suggestions = random.choice([2, 3])
            selected_maintenance = random.sample(self.clear_face_maintenance, num_suggestions)
            treatment_text = self._format_multiple_treatments(selected_maintenance)
            return f"Ah! Your skin looks incredibly healthy and clear. To maintain this, {transition.strip().lower()} {treatment_text}."

        symptom_descriptions = []
        all_treatments = []

        for symptom, confidence in diagnoses:
            formatted_symp = symptom.replace("_", " ")
            sev_level = self._determine_severity(confidence)
            adj = random.choice(self.severity_adjectives[sev_level])
            symptom_descriptions.append(f"{adj}{formatted_symp}")
            
            # Pull 1 or 2 treatments per symptom to avoid overwhelming the user
            num_treatments = random.choice([1, 2])
            treatments = random.sample(self.treatments[symptom][sev_level], num_treatments)
            all_treatments.extend(treatments)

        if len(symptom_descriptions) > 1:
            joined_symptoms = " alongside ".join(symptom_descriptions)
        else:
            joined_symptoms = symptom_descriptions[0]

        treatment_text = self._format_multiple_treatments(all_treatments)

        final_sentence = f"{intro}{joined_symptoms}.{transition}{treatment_text}{outro}"
        return final_sentence.capitalize()

class ImprovementObserver:
    def __init__(self):
        # 1. SUCCESS STATE
        self.success_intros = [
            "Incredible work! ", "Congratulations! ", "Excellent progress! ", 
            "I'm seeing fantastic results! ", "Your consistency is paying off! ",
            "The data doesn't lie—great job! "
        ]
        
        # 2. PLATEAU STATE
        self.pivot_intros = [
            "It looks like the topical treatments haven't made a major dent over the last 10 days. ",
            "Skin can be stubborn, and it seems our initial routine hasn't induced a major difference yet. ",
            "I'm not seeing a significant reduction in the past 10 days, but don't worry. ",
            "We aren't seeing the drop we wanted just yet. ",
            "Topicals can only do so much, and we've plateaued a bit here. ",
            "The progress is a little slower than anticipated. "
        ]

        # 3. WORSENED STATE (SOS)
        self.worsened_intros = [
            "I am looking at the data, and it appears the symptoms have unfortunately intensified. ",
            "Oh no, it seems the current routine is irritating your skin further. ",
            "I'm concerned to see that the condition has actually worsened over the last 10 days. ",
            "Stop your active treatments immediately. The scan shows increased irritation. ",
            "It looks like your skin barrier is reacting poorly to the regimen. "
        ]

        self.holistic_interventions = {
            "acne": [
                "cutting whey protein and refined sugars from your diet",
                "switching to a fresh silk pillowcase every two days",
                "evaluating your gut health with a daily probiotic",
                "tracking dairy intake, as it can heavily trigger hormonal breakouts",
                "drinking spearmint tea daily to help balance androgens"
            ],
            "darkspots": [
                "ensuring you are reapplying SPF 50 every exactly two hours",
                "wearing a UPF 50 wide-brimmed hat to physically block UV rays",
                "increasing your dietary intake of Vitamin C rich fruits",
                "checking for underlying hormonal imbalances like Melasma",
                "taking an oral Polypodium Leucotomos antioxidant supplement"
            ],
            "puffy_eyes": [
                "sleeping with your head slightly elevated on an extra pillow",
                "drastically reducing your evening sodium and salt intake",
                "drinking an extra liter of water to flush out retained fluids",
                "getting a strict 8 hours of sleep to reduce cortisol",
                "reducing screen time an hour before bed to prevent eye strain and fluid pooling"
            ],
            "wrinkles": [
                "increasing your intake of Omega-3 fatty acids like salmon or walnuts",
                "focusing on deep hydration by drinking 3 liters of water daily",
                "avoiding sleeping on your side to prevent compression lines",
                "incorporating bone broth or collagen supplements into your diet",
                "managing daily stress to prevent premature cellular aging"
            ]
        }

        # SOS Protocols for worsened conditions
        self.sos_interventions = {
            "acne": [
                "stop all chemical exfoliants and harsh washes immediately",
                "strip your routine down to just a gentle cleanser and a basic moisturizer",
                "apply a soothing Cicaplast baume to repair your skin barrier",
                "consult a dermatologist to prevent any potential deep scarring"
            ],
            "darkspots": [
                "stop all brightening acids, as this might be irritation-induced hyperpigmentation",
                "focus entirely on barrier repair and strict SPF application",
                "apply a healing ointment to any raw or sensitive areas",
                "see a professional to rule out a chemical burn or allergic reaction"
            ],
            "puffy_eyes": [
                "stop all eye creams immediately, as this strongly indicates contact dermatitis or an allergy",
                "take an over-the-counter antihistamine",
                "apply a cool, damp cloth with absolutely no active ingredients",
                "consult a doctor if the swelling persists or affects your vision"
            ],
            "wrinkles": [
                "stop all retinoids and retinols immediately—your skin barrier is compromised",
                "switch to a heavy, fragrance-free ceramide cream",
                "avoid any physical rubbing or facial massages until the skin heals",
                "use a gentle, milky cleanser instead of foaming washes"
            ]
        }

    def evaluate_10_day_trial(self, diagnoses: list, status: str) -> str:
        """
        diagnoses: list of current symptoms, e.g., ["acne", "darkspots"]
        status: "improved", "plateau", or "worsened"
        """
        if "clear_face" in diagnoses:
            return "Your skin is remaining perfectly clear! Keep doing exactly what you are doing."

        symptom_str = " and ".join([s.replace("_", " ") for s in diagnoses])

        # 1. THE SUCCESS PATH
        if status == "improved":
            intro = random.choice(self.success_intros)
            return f"{intro}Over the past 10 days, your {symptom_str} has visibly reduced. The current treatment is working perfectly—stay consistent with this routine."

        # 2. THE WORSENED (SOS) PATH
        if status == "worsened":
            intro = random.choice(self.worsened_intros)
            sos_advice = []
            for symptom in diagnoses:
                if symptom in self.sos_interventions:
                    # Grab 2 critical SOS steps
                    advice_picks = random.sample(self.sos_interventions[symptom], 2)
                    sos_advice.extend(advice_picks)
            
            # Format the emergency advice
            if len(sos_advice) > 1:
                joined_sos = ", and ".join(sos_advice[:-1]) + ", and " + sos_advice[-1]
            else:
                joined_sos = sos_advice[0]

            return f"{intro}Please {joined_sos}."

        # 3. THE PLATEAU PATH
        intro = random.choice(self.pivot_intros)
        lifestyle_advice = []
        for symptom in diagnoses:
            if symptom in self.holistic_interventions:
                advice_picks = random.sample(self.holistic_interventions[symptom], random.choice([1, 2]))
                lifestyle_advice.extend(advice_picks)

        if len(lifestyle_advice) > 1:
            joined_advice = ", and ".join(lifestyle_advice[:-1]) + ", and " + lifestyle_advice[-1]
        else:
            joined_advice = lifestyle_advice[0]

        return f"{intro}Let's pivot away from just using skincare products. For the next week, I strongly suggest {joined_advice}."

# --- MAIN TEST HARNESS ---
if __name__ == "__main__":
    recommender = SkincareRecommender() # Assuming this is still in your file
    observer = ImprovementObserver()
    
    print("\n" + "="*50)
    print(" 🏥 DERMASCAN AI: 3-STATE FEEDBACK LOOP TEST 🏥 ")
    print("="*50)

    # Test 1: 10-Day Follow-Up (Success)
    print("\n--- TEST 1: DAY 10 (IMPROVED ON PUFFY EYES) ---")
    print(observer.evaluate_10_day_trial(["puffy_eyes"], status="improved"))

    # Test 2: 10-Day Follow-Up (Plateau)
    print("\n--- TEST 2: DAY 10 (PLATEAU ON ACNE) ---")
    print(observer.evaluate_10_day_trial(["acne"], status="plateau"))

    # Test 3: 10-Day Follow-Up (Worsened SOS Protocol)
    print("\n--- TEST 3: DAY 10 (WORSENED ON DARK SPOTS & WRINKLES) ---")
    print(observer.evaluate_10_day_trial(["darkspots", "wrinkles"], status="worsened"))
    
    print("\n" + "="*50)
