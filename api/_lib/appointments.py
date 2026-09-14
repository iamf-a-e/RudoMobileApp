"""
appointments.py — Appointment booking & rescheduling flow for Rudo
===================================================================
Drop this file into api/_lib/ alongside engine.py.

This module adds a complete, stateful appointment booking/rescheduling flow
to handle_turn(). It is designed to slot directly into the existing engine
without changing any other file — just import and call it from handle_turn().

USAGE in engine.py
------------------
1. At the top of engine.py, add:
       from _lib.appointments import (
           APPOINTMENT_KEYWORDS,
           handle_appointment_turn,
           greet_booking_start,
       )

2. In handle_turn(), BEFORE the shop flow block, add:

       # ---- appointment booking / rescheduling ----
       if (
           _contains_signal(prompt_lower, APPOINTMENT_KEYWORDS)
           or step.startswith("appt_")
       ):
           handle_appointment_turn(user_id, message, state, out)
           return

That's it. The rest of engine.py is unchanged.

STATE MACHINE
-------------
appt_collect_date      → ask for preferred date
appt_collect_time      → ask for preferred time
appt_collect_type      → ask what kind of visit (ANC, scan, general, etc.)
appt_collect_clinic    → ask which clinic / health worker (optional)
appt_confirm           → show summary and ask "yes / no"
appt_done              → confirmed, save to Redis, return to main_menu

RESCHEDULING
------------
If the user mentions "reschedule", "change appointment", etc., the flow
branches to appt_reschedule_confirm, clears the old booking, then restarts
the same collect steps with a note that this is a change.

LANGUAGES SUPPORTED
-------------------
All seven languages in SUPPORTED_LANGUAGES: english, shona, ndebele,
chinyanja, bemba, tonga, lozi.
"""

import json
import logging
import re
from datetime import datetime

# ─────────────────────────────────────────────
#  Trigger keywords (injected into handle_turn)
# ─────────────────────────────────────────────

APPOINTMENT_KEYWORDS = [
    # English
    "appointment", "book", "booking", "schedule", "visit", "clinic", "reschedule",
    "change appointment", "cancel appointment", "see a doctor", "see doctor",
    "health worker", "antenatal", "anc", "prenatal visit", "check-up", "checkup",
    # Shona
    "bvunza chiremba", "chikwama", "kuona chiremba", "kuchekwa", "murapi",
    "buka appointment", "buka", "shandura appointment",
    # Ndebele
    "bona udokotela", "hlela ukuhlangana", "hlela", "shintsha",
    # Chinyanja
    "onana ndi dokotala", "buku", "chitukuko", "chipatala",
    # Bemba
    "mwene ubwanga", "panga appointment", "dokota",
    # Tonga
    "bona silalikani", "botelela", "dokotela",
    # Lozi
    "bona ngaka", "bulela", "layela",
]

RESCHEDULE_KEYWORDS = [
    "reschedule", "change appointment", "change my appointment", "move appointment",
    "different time", "different date", "shandura appointment", "shintsha",
    "shandura", "badilisha",
]

# ─────────────────────────────────────────────
#  Multilingual string tables
# ─────────────────────────────────────────────

STRINGS = {
    "ask_date": {
        "english": "📅 What date would you like your appointment? (e.g. Monday 23 June, or 23/06)",
        "shona": "📅 Unoda zuva ripi reappointment yako? (somuenzaniso: Muvhuro 23 Chikumi, kana 23/06)",
        "ndebele": "📅 Ufuna usuku luni lokuhlangana? (isibonelo: UMsombuluko 23 Nhlangulana, kumbe 23/06)",
        "chinyanja": "📅 Mukufuna tsiku lanji lodzera? (mwachitsanzo: Lachisanu 23 Juni, kapena 23/06)",
        "bemba": "📅 Mufwaya ubushiku nshi bwa appointment? (somuenzaniso: Palichimo 23 Juni, nangu 23/06)",
        "tonga": "📅 Muyanda buzuba nzi bwa kubonana? (somuenzaniso: Cimponde 23 Juni, naa 23/06)",
        "lozi": "📅 Mu bata lizazi lifi la appointment? (mutala: Museli 23 Juni, kamba 23/06)",
    },
    "ask_time": {
        "english": "🕐 What time works best for you? (e.g. 9am, 10:30, afternoon)",
        "shona": "🕐 Nguva ipi inokushandira? (somuenzaniso: 9am, 10:30, masikati)",
        "ndebele": "🕐 Isikhathi sini esikufanelayo? (isibonelo: 9am, 10:30, ntambama)",
        "chinyanja": "🕐 Nthawi yotani imakugwirizani? (mwachitsanzo: 9am, 10:30, masana)",
        "bemba": "🕐 Inshita nshi iyamipela? (somuenzaniso: 9am, 10:30, icungulo)",
        "tonga": "🕐 Nthawi nzi iyakumugwasya? (somuenzaniso: 9am, 10:30, musonde)",
        "lozi": "🕐 Nako yafi e ku mi swanela? (mutala: 9am, 10:30, nako ya minyanu)",
    },
    "ask_type": {
        "english": "🏥 What type of visit is this?\n1. Antenatal check-up (ANC)\n2. Ultrasound / scan\n3. General consultation\n4. Cervical cancer screening\n5. Other",
        "shona": "🏥 Kuona kwerudzi rwei?\n1. Kuona kwepamuviri (ANC)\n2. Ultrasound / scan\n3. Kukurukurirana nachiremba\n4. Kuongorwa kwegonorrhoea / cancer yechibereko\n5. Zvimwe",
        "ndebele": "🏥 Uhlobo luni lwakubona?\n1. Ukubona kwesisu (ANC)\n2. I-ultrasound / scan\n3. Ukuxoxa lodokotela\n4. Ukuhlolwa kwumhlaza wesibeleko\n5. Okunye",
        "chinyanja": "🏥 Mtundu wanji wa kuonana?\n1. Kuona kwa mimba (ANC)\n2. Ultrasound / scan\n3. Kukambirana ndi dokotala\n4. Kuwunikidwa kansa ya mchombo\n5. Zina",
        "bemba": "🏥 Ubwanga bwashani bwa kubonana?\n1. Kuona kwa pamimba (ANC)\n2. Ultrasound / scan\n3. Kukambana na dokota\n4. Ukwelekanyishiwa kansa ya munda\n5. Fimbi",
        "tonga": "🏥 Mulimo nzi wa kubonana?\n1. Kubona kwa nhumbu (ANC)\n2. Ultrasound / scan\n3. Kukambana a silalikani\n4. Kulangilizigwa kansa ya munda\n5. Chimwi",
        "lozi": "🏥 Muhato mañi wa kubonana?\n1. Kubona kwa nhumbu (ANC)\n2. Ultrasound / scan\n3. Kukamba le ngaka\n4. Kubaliwa kansa ya mbo\n5. Se siñwi",
    },
    "ask_clinic": {
        "english": "📍 Do you have a preferred clinic or health worker? (Type their name, or say 'any' / 'no preference')",
        "shona": "📍 Une chipatara kana murapi waunoda? (Nyora zita ravo, kana uti 'chero' / 'hapana sarudzo')",
        "ndebele": "📍 Ulesikhungo sezempilo noma isisebenzi sezempilo ofuna sona? (Bhala igama laso, noma uthi 'noma yiliphi' / 'anginandaba')",
        "chinyanja": "📍 Muli ndi chipatala kapena wothandiza wazachipatala amene mukufuna? (Lembani dzina lawo, kapena muneneli 'aliyense' / 'palibe kusankha')",
        "bemba": "📍 Muli na chipatala nangu umuceshi wa mwenge mucifwayo? (Temwa ishina lyabo, nangu mwebe ati 'ifi konse' / 'tafinankwe icisankwa')",
        "tonga": "📍 Muli a chipatala naa silalikani ncomuyanda? (Lembanya zina lyabo, naa mube kuti 'ulayanda' / 'tabuli kusankha')",
        "lozi": "📍 Mu na sipatela kamba muoki wa bupilo mo mu bata? (Ñola libizo la bona, kamba mu bulele 'ufi ni ufi' / 'ha ni si na takazo')",
    },
    "confirm_prompt": {
        "english": "✅ Please confirm your appointment:\n\n📅 Date: {date}\n🕐 Time: {time}\n🏥 Type: {visit_type}\n📍 Clinic/Provider: {clinic}\n\nReply *yes* to confirm or *no* to change.",
        "shona": "✅ Simbidza appointment yako:\n\n📅 Zuva: {date}\n🕐 Nguva: {time}\n🏥 Rudzi: {visit_type}\n📍 Chipatara/Murapi: {clinic}\n\nDzosera *hongu* kusimbidza kana *kwete* kuchinja.",
        "ndebele": "✅ Qinisekisa ukuhlangana kwakho:\n\n📅 Usuku: {date}\n🕐 Isikhathi: {time}\n🏥 Uhlobo: {visit_type}\n📍 Ikhliniki/Umhlinzeki: {clinic}\n\nPhendula *yebo* ukuqinisekisa noma *hatshi* ukushintsha.",
        "chinyanja": "✅ Tsimikizani nthawi yanu ya kuonana:\n\n📅 Tsiku: {date}\n🕐 Nthawi: {time}\n🏥 Mtundu: {visit_type}\n📍 Chipatala/Wopereka: {clinic}\n\nYankha *inde* kutsimikiza kapena *ayi* kusintha.",
        "bemba": "✅ Pokeleni appointment yenu:\n\n📅 Ubushiku: {date}\n🕐 Inshita: {time}\n🏥 Ubwanga: {visit_type}\n📍 Chipatala/Umuceshi: {clinic}\n\nYasuka *ee* ukupokelela nangu *awe* ukusankula.",
        "tonga": "✅ Tompolezya appointment yanu:\n\n📅 Buzuba: {date}\n🕐 Nthawi: {time}\n🏥 Mulimo: {visit_type}\n📍 Chipatala/Silalikani: {clinic}\n\nPandula *ee* kutompolezya naa *ayi* kulemba kabili.",
        "lozi": "✅ Tiisa appointment ya hao:\n\n📅 Lizazi: {date}\n🕐 Nako: {time}\n🏥 Muhato: {visit_type}\n📍 Sipatela/Muoki: {clinic}\n\nAraba *inde* ku tiisa kamba *batili* ku cincisa.",
    },
    "confirmed": {
        "english": "🎉 Your appointment has been booked!\n\n📅 {date} at {time}\n🏥 {visit_type}\n📍 {clinic}\n\nWe'll send you a reminder. Is there anything else I can help you with?",
        "shona": "🎉 Appointment yako yakabukwa!\n\n📅 {date} pa {time}\n🏥 {visit_type}\n📍 {clinic}\n\nTichakutumira chirangaridzo. Kune chimwe chandingakubatsire nacho?",
        "ndebele": "🎉 Ukuhlangana kwakho kuqinisekisiwe!\n\n📅 {date} ngo {time}\n🏥 {visit_type}\n📍 {clinic}\n\nSizakuthumela isikhumbuzo. Kukhona okunye engingakusiza ngakho?",
        "chinyanja": "🎉 Nthawi yanu ya kuonana yabukidwa!\n\n📅 {date} pa {time}\n🏥 {visit_type}\n📍 {clinic}\n\nTizatumiza chikumbutso. Kuli china chomwe ndingakuthandizeni nacho?",
        "bemba": "🎉 Appointment yenu yapangwa!\n\n📅 {date} pa {time}\n🏥 {visit_type}\n📍 {clinic}\n\nTukatuma ukukumbushako. Kuli fimbi fikwete namwafwa?",
        "tonga": "🎉 Appointment yanu yatompolezgwa!\n\n📅 {date} pa {time}\n🏥 {visit_type}\n📍 {clinic}\n\nTizatuma kukumbushizya. Kuli chimwi cho ngatamugwasye?",
        "lozi": "🎉 Appointment ya hao i tiisizwe!\n\n📅 {date} fa {time}\n🏥 {visit_type}\n📍 {clinic}\n\nLu ka mi luma sikombuzo. Kuna se siñwi mo ni ka mi thusa?",
    },
    "cancelled": {
        "english": "Okay, let's start over. What date would you like for your appointment?",
        "shona": "Zvakanaka, ngatitangei patsva. Unoda zuva ripi reappointment yako?",
        "ndebele": "Kulungile, asiqale kabusha. Ufuna usuku luni lokuhlangana?",
        "chinyanja": "Chabwino, tiyambenso. Mukufuna tsiku lanji lodzera?",
        "bemba": "Cishinka, natangile panono. Mufwaya ubushiku nshi bwa appointment?",
        "tonga": "Kabotu, twatande kabili. Muyanda buzuba nzi bwa kubonana?",
        "lozi": "Kulukile, lu kalise hape. Mu bata lizazi lifi la appointment?",
    },
    "reschedule_start": {
        "english": "I'll help you reschedule. Let me clear your previous appointment. What new date would you like?",
        "shona": "Ndichakubatsira kushandura. Ndichabvisa appointment yako yekare. Unoda zuva ipi idzva?",
        "ndebele": "Ngizakusiza ukushintsha. Ngizasusa ukuhlangana kwakho kwangaphambili. Ufuna usuku luni olusha?",
        "chinyanja": "Ndikuthandizeni kusintha. Ndifufuta nthawi yanu yakale. Mukufuna tsiku lanji latsopano?",
        "bemba": "Namwafwa kusankula. Nkafutako appointment yenu yakale. Mufwaya ubushiku nshi bushipe?",
        "tonga": "Ngatamugwasye kusintha. Ndiyowole appointment yanu yakale. Muyanda buzuba nzi bushipe?",
        "lozi": "Ni ka mi thusa ku cincisa. Ni ka fula appointment ya hao ya kwa makalelo. Mu bata lizazi lifi lisha?",
    },
    "invalid_input": {
        "english": "I didn't quite catch that. Could you please try again?",
        "shona": "Handina kunzwisisa. Ungada kuedza zvakare?",
        "ndebele": "Angizwisisanga. Ungayama ukuphinda?",
        "chinyanja": "Sindinamve bwino. Mungayesa kuyankhananso?",
        "bemba": "Nshaumfwa bwino. Mungayesa kuyansula kabili?",
        "tonga": "Nsikamvwa bwino. Mungayesa kupandula kabili?",
        "lozi": "Ha ni utwi hande. Mu ka lika ku araba hape?",
    },
}

VISIT_TYPES = {
    "1": {"english": "Antenatal check-up (ANC)", "shona": "Kuona kwepamuviri (ANC)", "ndebele": "Ukubona kwesisu (ANC)",
          "chinyanja": "Kuona kwa mimba (ANC)", "bemba": "Kuona kwa pamimba (ANC)",
          "tonga": "Kubona kwa nhumbu (ANC)", "lozi": "Kubona kwa nhumbu (ANC)"},
    "2": {"english": "Ultrasound / Scan", "shona": "Ultrasound / Scan", "ndebele": "I-ultrasound / Scan",
          "chinyanja": "Ultrasound / Scan", "bemba": "Ultrasound / Scan",
          "tonga": "Ultrasound / Scan", "lozi": "Ultrasound / Scan"},
    "3": {"english": "General consultation", "shona": "Kukurukurirana nachiremba", "ndebele": "Ukuxoxa lodokotela",
          "chinyanja": "Kukambirana ndi dokotala", "bemba": "Kukambana na dokota",
          "tonga": "Kukambana a silalikani", "lozi": "Kukamba le ngaka"},
    "4": {"english": "Cervical cancer screening", "shona": "Kuongorwa kwecancer yechibereko",
          "ndebele": "Ukuhlolwa kwumhlaza wesibeleko", "chinyanja": "Kuwunikidwa kansa ya mchombo",
          "bemba": "Ukwelekanyishiwa kansa ya munda", "tonga": "Kulangilizigwa kansa ya munda",
          "lozi": "Kubaliwa kansa ya mbo"},
    "5": {"english": "Other", "shona": "Zvimwe", "ndebele": "Okunye",
          "chinyanja": "Zina", "bemba": "Fimbi", "tonga": "Chimwi", "lozi": "Se siñwi"},
}

YES_WORDS = ["yes", "yeah", "yep", "please", "ehe", "hongu", "inde", "yebo", "ee", "inde", "eya"]
NO_WORDS = ["no", "nah", "nope", "hapana", "kwete", "aiwa", "cha", "ayi", "not really", "hatshi", "awe", "batili"]


def _t(key, lang):
    """Get a string for the given key in the given language, fallback to English."""
    return STRINGS[key].get(lang, STRINGS[key]["english"])


def _contains(text, words):
    text_l = text.lower()
    return any(w in text_l for w in words)


def _save_appointment(redis_client, user_id, state):
    """Persist the confirmed appointment to Redis."""
    if not redis_client:
        return
    try:
        appt = {
            "user_id": user_id,
            "date": state.get("appt_date"),
            "time": state.get("appt_time"),
            "visit_type": state.get("appt_visit_type"),
            "clinic": state.get("appt_clinic"),
            "status": "confirmed",
            "booked_at": datetime.now().isoformat(),
        }
        key = f"appointments:{user_id}:{datetime.now().strftime('%Y%m%d%H%M%S')}"
        redis_client.set(key, json.dumps(appt), ex=60 * 60 * 24 * 90)  # 90 days
        # Also store as the user's "current appointment" for quick lookup
        redis_client.set(f"current_appointment:{user_id}", json.dumps(appt), ex=60 * 60 * 24 * 90)
        logging.info(f"Appointment saved for {user_id}: {appt}")
    except Exception as e:
        logging.error(f"Error saving appointment for {user_id}: {e}")


def _clear_appointment_state(state):
    """Wipe appointment-related keys from state, keeping everything else."""
    for key in ["appt_date", "appt_time", "appt_visit_type", "appt_clinic", "appt_is_reschedule"]:
        state.pop(key, None)


def _resolve_visit_type(message, lang):
    """
    Return the localised visit type string from a digit choice or free text.
    Returns None if nothing matched.
    """
    msg = message.strip()
    # Digit choice
    if msg in VISIT_TYPES:
        return VISIT_TYPES[msg].get(lang, VISIT_TYPES[msg]["english"])
    # Free-text match against English keys
    msg_l = msg.lower()
    for num, labels in VISIT_TYPES.items():
        for label in labels.values():
            if label.lower() in msg_l or msg_l in label.lower():
                return labels.get(lang, labels["english"])
    # Short keyword matches
    if any(kw in msg_l for kw in ["anc", "antenatal", "prenatal", "pamuviri", "mimba", "nhumbu"]):
        return VISIT_TYPES["1"].get(lang, VISIT_TYPES["1"]["english"])
    if any(kw in msg_l for kw in ["scan", "ultrasound"]):
        return VISIT_TYPES["2"].get(lang, VISIT_TYPES["2"]["english"])
    if any(kw in msg_l for kw in ["consult", "general", "chiremba", "dokot"]):
        return VISIT_TYPES["3"].get(lang, VISIT_TYPES["3"]["english"])
    if any(kw in msg_l for kw in ["cervical", "cancer", "screen", "chibereko", "mchombo", "munda"]):
        return VISIT_TYPES["4"].get(lang, VISIT_TYPES["4"]["english"])
    return None


def handle_appointment_turn(user_id, message, state, out, redis_client=None):
    """
    Stateful appointment booking/rescheduling handler.

    Call this from handle_turn() when APPOINTMENT_KEYWORDS trigger or
    state["step"].startswith("appt_").

    Parameters
    ----------
    user_id     : str
    message     : str   — raw user message
    state       : dict  — mutable user state from Redis
    out         : list  — append reply strings here (same as handle_turn)
    redis_client: Redis — pass the module-level redis_client from engine.py
    """
    lang = state.get("language", "english")
    step = state.get("step", "main_menu")
    msg_l = message.lower().strip()

    # ── RESCHEDULE entry point ─────────────────────────────────────────────
    if _contains(msg_l, RESCHEDULE_KEYWORDS) and not step.startswith("appt_"):
        _clear_appointment_state(state)
        state["appt_is_reschedule"] = True
        state["step"] = "appt_collect_date"
        out.append(_t("reschedule_start", lang))
        return

    # ── FRESH BOOKING entry point ──────────────────────────────────────────
    if not step.startswith("appt_"):
        _clear_appointment_state(state)
        state["appt_is_reschedule"] = False
        state["step"] = "appt_collect_date"
        out.append(_t("ask_date", lang))
        return

    # ══════════════════════════════════════════════════════════════════════
    #  STATE MACHINE
    # ══════════════════════════════════════════════════════════════════════

    # ── appt_collect_date ─────────────────────────────────────────────────
    if step == "appt_collect_date":
        # Accept anything that looks like a date: digits, day/month names, etc.
        # We store whatever the user typed — a proper calendar integration
        # would validate/parse here, but for MVP we trust the user's input.
        if len(msg_l) >= 3:  # at least 3 chars — "Mon", "5/6", etc.
            state["appt_date"] = message.strip()
            state["step"] = "appt_collect_time"
            out.append(_t("ask_time", lang))
        else:
            out.append(_t("invalid_input", lang) + "\n\n" + _t("ask_date", lang))
        return

    # ── appt_collect_time ─────────────────────────────────────────────────
    if step == "appt_collect_time":
        # Accept "9am", "09:00", "afternoon", "masikati", etc.
        if len(msg_l) >= 2:
            state["appt_time"] = message.strip()
            state["step"] = "appt_collect_type"
            out.append(_t("ask_type", lang))
        else:
            out.append(_t("invalid_input", lang) + "\n\n" + _t("ask_time", lang))
        return

    # ── appt_collect_type ─────────────────────────────────────────────────
    if step == "appt_collect_type":
        visit_type = _resolve_visit_type(message, lang)
        if visit_type:
            state["appt_visit_type"] = visit_type
            state["step"] = "appt_collect_clinic"
            out.append(_t("ask_clinic", lang))
        else:
            # If we still can't parse after a re-prompt, just accept free text
            if state.get("appt_type_retry"):
                state["appt_visit_type"] = message.strip() if message.strip() else "General"
                state.pop("appt_type_retry", None)
                state["step"] = "appt_collect_clinic"
                out.append(_t("ask_clinic", lang))
            else:
                state["appt_type_retry"] = True
                out.append(_t("invalid_input", lang) + "\n\n" + _t("ask_type", lang))
        return

    # ── appt_collect_clinic ───────────────────────────────────────────────
    if step == "appt_collect_clinic":
        NO_PREF = ["any", "no preference", "chero", "hapana", "hatshi", "ifi konse",
                   "tafinankwe", "ulayanda", "ufi ni ufi", "aliyense", "noma yiliphi"]
        if _contains(msg_l, NO_PREF) or msg_l in ["any", "no", "none", ""]:
            state["appt_clinic"] = {"english": "Any available", "shona": "Chero chinotowika",
                                    "ndebele": "Noma yiliphi etholakayo", "chinyanja": "Aliyense wopezeka",
                                    "bemba": "Ifi konse ifyapezeka", "tonga": "Ulayanda wapezeka",
                                    "lozi": "Ufi ni ufi wa fumaneha"}.get(lang, "Any available")
        else:
            state["appt_clinic"] = message.strip()

        # Move to confirmation
        state["step"] = "appt_confirm"
        confirm_msg = _t("confirm_prompt", lang).format(
            date=state.get("appt_date", "?"),
            time=state.get("appt_time", "?"),
            visit_type=state.get("appt_visit_type", "?"),
            clinic=state.get("appt_clinic", "?"),
        )
        out.append(confirm_msg)
        return

    # ── appt_confirm ──────────────────────────────────────────────────────
    if step == "appt_confirm":
        if _contains(msg_l, YES_WORDS):
            # ✅ CONFIRMED
            _save_appointment(redis_client, user_id, state)
            confirmed_msg = _t("confirmed", lang).format(
                date=state.get("appt_date", "?"),
                time=state.get("appt_time", "?"),
                visit_type=state.get("appt_visit_type", "?"),
                clinic=state.get("appt_clinic", "?"),
            )
            out.append(confirmed_msg)
            _clear_appointment_state(state)
            state["step"] = "main_menu"
            state["topic"] = None

        elif _contains(msg_l, NO_WORDS):
            # ❌ User wants to change — restart the collect flow
            _clear_appointment_state(state)
            state["step"] = "appt_collect_date"
            out.append(_t("cancelled", lang))

        else:
            # Re-show the confirmation prompt
            confirm_msg = _t("confirm_prompt", lang).format(
                date=state.get("appt_date", "?"),
                time=state.get("appt_time", "?"),
                visit_type=state.get("appt_visit_type", "?"),
                clinic=state.get("appt_clinic", "?"),
            )
            out.append(_t("invalid_input", lang) + "\n\n" + confirm_msg)
        return

    # ── Safety net: unknown appt_ step ───────────────────────────────────
    logging.warning(f"Unknown appointment step '{step}' for user {user_id} — resetting")
    _clear_appointment_state(state)
    state["step"] = "appt_collect_date"
    out.append(_t("ask_date", lang))


def greet_booking_start(lang):
    """Convenience greeting when user says 'book appointment' as first message."""
    return {
        "english": "I'll help you book an appointment. Let's get the details. 😊",
        "shona": "Ndichakubatsira kubhuka appointment. Ngatiwane ruzivo. 😊",
        "ndebele": "Ngizakusiza ukubhukha ukuhlangana. Ake sithole imininingwane. 😊",
        "chinyanja": "Ndikuthandizeni kubukitsa nthawi. Tiyitane zambiri. 😊",
        "bemba": "Namwafwa kupanga appointment. Natwashibe fimbi. 😊",
        "tonga": "Ngatamugwasye kubukitsa kubonana. Natwezyenge zambiri. 😊",
        "lozi": "Ni ka mi thusa ku buka appointment. Lu be ni taluso. 😊",
    }.get(lang, "I'll help you book an appointment. Let's get the details. 😊")
