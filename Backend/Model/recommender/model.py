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
        self.success_intros = [
            "Incredible work! ", "Congratulations! ", "Excellent progress! ", 
            "I'm seeing fantastic results! ", "Your consistency is paying off! ",
            "The data doesn't lie—great job! "
        ]
        
        self.pivot_intros = [
            "It looks like the topical treatments haven't made a major dent over the last 10 days. ",
            "Skin can be stubborn, and it seems our initial routine hasn't induced a major difference yet. ",
            "I'm not seeing a significant reduction in the past 10 days, but don't worry. ",
            "We aren't seeing the drop we wanted just yet. ",
            "Topicals can only do so much, and we've plateaued a bit here. ",
            "The progress is a little slower than anticipated. "
        ]

        self.holistic_interventions = {
            "acne": [
                "cutting whey protein and refined sugars from your diet",
                "switching to a fresh silk pillowcase every two days",
                "evaluating your gut health with a daily probiotic",
                "tracking dairy intake, as it can heavily trigger hormonal breakouts",
                "drinking spearmint tea daily to help balance androgens",
                "changing your face towel every single day",
                "wiping down your smartphone screen with alcohol daily",
                "switching to a non-comedogenic, oil-free laundry detergent",
                "actively managing stress levels, as cortisol spikes can trigger excess sebum",
                "changing out of sweaty workout clothes and showering immediately"
            ],
            "darkspots": [
                "ensuring you are reapplying SPF 50 every exactly two hours",
                "wearing a UPF 50 wide-brimmed hat to physically block UV rays",
                "increasing your dietary intake of Vitamin C rich fruits",
                "checking for underlying hormonal imbalances like Melasma",
                "strictly avoiding any picking or friction on your face",
                "taking an oral Polypodium Leucotomos antioxidant supplement",
                "applying SPF even when indoors near windows",
                "completely avoiding dry saunas or hot yoga, as ambient heat triggers melasma",
                "avoiding harsh physical scrubs that cause micro-tears and hyperpigmentation"
            ],
            "puffy_eyes": [
                "sleeping with your head slightly elevated on an extra pillow",
                "drastically reducing your evening sodium and salt intake",
                "drinking an extra liter of water to flush out retained fluids",
                "getting a strict 8 hours of sleep to reduce cortisol",
                "treating seasonal allergies with an over-the-counter antihistamine",
                "eliminating alcohol consumption at least 4 hours before bed",
                "using an ice roller on your face for 5 minutes every morning",
                "getting a blood test to check for underlying thyroid issues",
                "reducing screen time an hour before bed to prevent eye strain and fluid pooling"
            ],
            "wrinkles": [
                "increasing your intake of Omega-3 fatty acids like salmon or walnuts",
                "focusing on deep hydration by drinking 3 liters of water daily",
                "avoiding sleeping on your side to prevent compression lines",
                "incorporating bone broth or collagen supplements into your diet",
                "managing daily stress to prevent premature cellular aging",
                "investing in a silk sleep mask to prevent micro-creasing around the eyes",
                "stopping all use of drinking straws to prevent perioral 'smoker's lines'",
                "incorporating a high-quality Vitamin C and Vitamin E supplement into your diet",
                "practicing facial massage to release chronic tension in the forehead and jaw",
                "strictly eliminating smoking or vaping, which degrades collagen rapidly"
            ]
        }

    def evaluate_10_day_trial(self, diagnoses: list, improvement_observed: bool) -> str:
        if "clear_face" in diagnoses:
            return "Your skin is remaining perfectly clear! Keep doing exactly what you are doing."

        if improvement_observed:
            intro = random.choice(self.success_intros)
            symptom_str = " and ".join([s.replace("_", " ") for s in diagnoses])
            return f"{intro}Over the past 10 days, your {symptom_str} has visibly reduced. The current treatment is working perfectly—stay consistent with this routine."

        intro = random.choice(self.pivot_intros)
        
        lifestyle_advice = []
        for symptom in diagnoses:
            if symptom in self.holistic_interventions:
                # Give 1 or 2 holistic tips per symptom to build a full plan
                advice_picks = random.sample(self.holistic_interventions[symptom], random.choice([1, 2]))
                lifestyle_advice.extend(advice_picks)

        if len(lifestyle_advice) > 1:
            joined_advice = ", and ".join(lifestyle_advice[:-1]) + ", and " + lifestyle_advice[-1]
        else:
            joined_advice = lifestyle_advice[0]

        return f"{intro}Let's pivot away from just using skincare products. For the next week, I strongly suggest {joined_advice}."

# --- MAIN TEST HARNESS ---
if __name__ == "__main__":
    recommender = SkincareRecommender()
    observer = ImprovementObserver()
    
    print("\n" + "="*50)
    print(" 🏥 DERMASCAN AI: CLINICAL NLP TEST HARNESS 🏥 ")
    print("="*50)

    # Test 1: Single Symptom
    print("\n--- TEST 1: SINGLE SYMPTOM (SEVERE ACNE) ---")
    print(recommender.generate_prescription([("acne", 94.5)]))

    # Test 2: Multi-Symptom
    print("\n--- TEST 2: MULTI-SYMPTOM (MODERATE WRINKLES & MINOR DARK SPOTS) ---")
    print(recommender.generate_prescription([("wrinkles", 72.0), ("darkspots", 55.0)]))

    # Test 3: Clear Face
    print("\n--- TEST 3: CLEAR FACE ---")
    print(recommender.generate_prescription([("clear_face", 99.0)]))

    # Test 4: 10-Day Follow-Up (Success)
    print("\n--- TEST 4: 10-DAY FOLLOW UP (SUCCESS ON PUFFY EYES) ---")
    print(observer.evaluate_10_day_trial(["puffy_eyes"], improvement_observed=True))

    # Test 5: 10-Day Follow-Up (Pivot/Failure)
    print("\n--- TEST 5: 10-DAY FOLLOW UP (PIVOT ON SEVERE ACNE) ---")
    print(observer.evaluate_10_day_trial(["acne"], improvement_observed=False))
    
    print("\n" + "="*50)