# -*- coding: utf-8 -*-
"""Generates data/eval_cases.jsonl."""
import json, io
from collections import Counter

C = []


def case(id, text, language, category, verdict, check_worthy, retrieves=None,
         in_corpus=False, note=""):
    C.append({
        "id": id, "text": text, "language": language, "category": category,
        "expect": {"verdict": verdict, "check_worthy": check_worthy,
                   "retrieves": retrieves},
        "in_corpus": in_corpus, "note": note,
    })


# --- in-corpus rumours, English -------------------------------------------
case("upi_cashback_en",
     "SBI is giving Rs 5,000 cashback to every customer today. Forward this message to 10 people to claim your reward.",
     "en", "scam", "false", True, "seed_upi_cashback", True,
     "The canonical forward-to-claim scam.")
case("upi_pin_refund_en",
     "You must enter your UPI PIN to receive the refund credited to your account.",
     "en", "scam", "false", True, "seed_upi_pin_refund", True,
     "A PIN is never needed to receive money.")
case("free_laptop_en",
     "The central government is distributing free laptops to all students under a new scheme.",
     "en", "govt", "false", True, "seed_free_laptop", True, "")
case("note_gps_en",
     "The Rs 2,000 note contains a GPS nano chip that lets the government track it.",
     "en", "govt", "false", True, "seed_2000_note_gps", True, "")
case("hot_water_en",
     "Drinking hot water every 15 minutes kills the coronavirus before it reaches the lungs.",
     "en", "health", "false", True, "seed_covid_hot_water", True, "")
case("cow_urine_en",
     "Cow urine cures cancer and doctors are hiding the evidence.",
     "en", "health", "false", True, "seed_cow_urine_cure", True, "")
case("polio_drops_en",
     "Polio drops given at government camps cause infertility in children.",
     "en", "health", "false", True, "seed_polio_drops", True,
     "Health rumour with a real vaccination-programme cost.")
case("vaccine_chip_en",
     "COVID-19 vaccines contain a microchip that tracks the person who received it.",
     "en", "health", "false", True, "seed_vaccine_microchip", True, "")
case("unesco_anthem_en",
     "UNESCO has declared Jana Gana Mana the best national anthem in the world.",
     "en", "misc", "false", True, "seed_unesco_anthem", True, "")
case("aadhaar_deadline_en",
     "Aadhaar-PAN linking is compulsory by tomorrow or your bank account will be frozen.",
     "en", "govt", "false", True, "seed_aadhaar_pan_deadline", True, "")

# --- in-corpus rumours, Devanagari Hindi ----------------------------------
case("hot_water_hi",
     "गर्म पानी पीने से कोरोना ठीक हो जाता है, हर 15 मिनट में पीते रहिए।",
     "hi", "health", "false", True, "seed_covid_hot_water", True, "")
case("free_laptop_hi",
     "सरकार सभी छात्रों को मुफ्त लैपटॉप दे रही है, लिंक पर रजिस्टर करें।",
     "hi", "govt", "false", True, "seed_free_laptop", True, "")
case("cow_urine_hi",
     "गोमूत्र कैंसर ठीक करता है और डॉक्टर इसे छिपा रहे हैं।",
     "hi", "health", "false", True, "seed_cow_urine_cure", True, "")
case("note_gps_hi",
     "2000 रुपये के नोट में जीपीएस नैनो चिप लगी है जिससे सरकार उसे ट्रैक करती है।",
     "hi", "govt", "false", True, "seed_2000_note_gps", True, "")
case("upi_cashback_hi",
     "SBI हर ग्राहक को 5,000 रुपये कैशबैक दे रहा है, यह संदेश 10 लोगों को भेजें।",
     "hi", "scam", "false", True, "seed_upi_cashback", True, "")

# --- in-corpus rumours, romanised Hinglish --------------------------------
# These are the cases backend/evidence/translit.py exists for. Before it,
# four of them retrieved nothing at all.
case("hot_water_hinglish",
     "Bhai suno, WHO ne kaha hai ki garam paani peene se corona theek ho jata hai.",
     "hinglish", "health", "false", True, "seed_covid_hot_water", True,
     "Retrieved nothing before translit.py.")
case("cow_urine_hinglish",
     "Gomutra cancer theek karta hai, doctor log ye baat chhupa rahe hain.",
     "hinglish", "health", "false", True, "seed_cow_urine_cure", True,
     "Retrieved nothing before translit.py.")
case("free_recharge_hinglish",
     "Jio apni salgirah par 3 mahine ka free recharge de raha hai, abhi claim karo.",
     "hinglish", "scam", "false", True, "seed_free_recharge", True,
     "Retrieved nothing before translit.py.")
case("aadhaar_hinglish",
     "Kal tak aadhaar pan link nahi kiya to bank khata hamesha ke liye band ho jayega.",
     "hinglish", "govt", "false", True, "seed_aadhaar_pan_deadline", True,
     "Retrieved nothing before translit.py.")
case("free_laptop_hinglish",
     "Sarkar ne sabhi students ko free laptop dene ka announcement kiya hai.",
     "hinglish", "govt", "false", True, "seed_free_laptop", True, "")
case("upi_cashback_hinglish",
     "SBI har grahak ko 5000 rupaye cashback de raha hai, ye message 10 logon ko forward karo.",
     "hinglish", "scam", "false", True, "seed_upi_cashback", True, "")
case("polio_hinglish",
     "Sarkari camp mein di jane wali polio drops se bachchon mein bandhyapan hota hai.",
     "hinglish", "health", "false", True, "seed_polio_drops", True, "")
case("note_gps_hinglish",
     "2000 ke note mein GPS chip lagi hai, sarkar sab track kar rahi hai.",
     "hinglish", "govt", "false", True, "seed_2000_note_gps", True, "")

# --- true claims ----------------------------------------------------------
case("repo_rate_true",
     "The Reserve Bank of India reduced the repo rate by 25 basis points at its monetary policy meeting.",
     "en", "true", "true", True, "seed_rbi_repo_cut_true", True,
     "The one corpus entry rated true. A checker that cannot say true is a censor.")

# --- out of corpus: the honest answer is unverified -----------------------
# Nothing in a 32-entry demo index can settle these. Calling any of them
# false would be the system inventing an answer.
case("oos_metro_fare", "The Delhi Metro has raised fares by 40 percent from next Monday.",
     "en", "govt", "unverified", True, None, False, "Plausible, unindexed.")
case("oos_exam_postponed", "The NEET examination has been postponed to December this year.",
     "en", "govt", "unverified", True, None, False, "")
case("oos_bank_merger", "HDFC Bank and Axis Bank will merge next quarter, the RBI has approved it.",
     "en", "money", "unverified", True, None, False, "")
case("oos_actor_death", "A well known Bollywood actor passed away this morning in Mumbai.",
     "en", "event", "unverified", True, None, False, "")
case("oos_hinglish_rumour",
     "Sarkar ne naya niyam banaya hai ki har ghar ko bijli bill mein 50 percent chhoot milegi.",
     "hinglish", "govt", "unverified", True, None, False,
     "Hinglish AND unindexed: must not be dragged onto a nearby seed entry.")
case("oos_hindi_rumour",
     "नई दिल्ली में कल से सभी स्कूल दो हफ्ते के लिए बंद कर दिए गए हैं।",
     "hi", "govt", "unverified", True, None, False, "")

# --- personal chat: not check-worthy, never accused ------------------------
case("chat_coming_home", "Hi bhai, kaise ho? Kal shaam ko ghar aa raha hoon.",
     "hinglish", "chat", "unverified", False, None, False, "")
case("chat_dinner", "Amma ne poocha tha ki tum dinner ke liye rukoge ya nahi.",
     "hinglish", "chat", "unverified", False, None, False, "")
case("chat_salt", "Namak lekar aana ghar aate waqt, khatam ho gaya hai.",
     "hinglish", "chat", "unverified", False, None, False,
     "Regression guard: transliteration pushed this to 0.459 against the "
     "salt-shortage rumour. See DECISIONS.md O1.")
case("chat_movie", "Bhai kal movie dekhne chalein kya, tickets book kar loon?",
     "hinglish", "chat", "unverified", False, None, False, "")
case("chat_birthday", "Happy birthday bhai, bahut bahut badhai ho aapko.",
     "hinglish", "chat", "unverified", False, None, False, "")
case("chat_office", "Aaj office mein bahut kaam tha, isliye reply nahi kar paya.",
     "hinglish", "chat", "unverified", False, None, False, "")
case("chat_meeting_en", "Please send me the meeting notes from yesterday before the standup.",
     "en", "chat", "unverified", False, None, False, "")
case("chat_match_en", "The cricket match was postponed due to rain in Mumbai last night.",
     "en", "chat", "unverified", False, None, False,
     "Reads as an event claim, but there is nothing to check it against.")

# --- benign true-ish statements that must never be called false -----------
case("benign_boiling_water",
     "Boiling water before drinking it reduces the risk of waterborne disease.",
     "en", "health", "unverified", True, None, False,
     "Adjacent to the hot-water rumour and actually sound. A false here is "
     "the worst failure the system can produce.")
case("benign_vaccine_safe",
     "COVID-19 vaccines were tested in clinical trials before being approved.",
     "en", "health", "unverified", True, None, False,
     "Sits next to the microchip entry in embedding space.")
case("benign_upi_safe",
     "You should never share your UPI PIN with anyone, including bank staff.",
     "en", "scam", "unverified", True, None, False,
     "Correct safety advice worded like the PIN scam it warns about.")


with io.open("data/eval_cases.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for c in C:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

print(len(C), "cases")
print("by language:", dict(Counter(c["language"] for c in C)))
print("by category:", dict(Counter(c["category"] for c in C)))
print("by expected verdict:", dict(Counter(c["expect"]["verdict"] for c in C)))
print("in corpus:", sum(c["in_corpus"] for c in C), "/", len(C))
assert len({c["id"] for c in C}) == len(C)
