import streamlit as st
import random
import base64
import streamlit.components.v1 as components
import pandas as pd
import requests
from datetime import datetime
from timezonefinder import TimezoneFinder
import pytz
import difflib
import matplotlib.pyplot as plt
#from Backend import (
#    load_data, get_coordinates_osm,
#    get_metno_weather, get_quiz_result,
#    show_destination, show_multiple_destinations, display_weather,
#    display_funny_theme_label, chatbot_step_handler,
#    handle_custom_location_query, get_user_theme_weights, add_to_bookmarks,
#    plot_quiz_scores, load_top10_data, render_top10_page, display_country_block,fetch_place_images,explain_dominant_style,
#    explain_all_styles,chatbot
#)

# ----------------- Data Loading & Preprocessing -----------------
def load_data():
    df = pd.read_csv("dataset_travel.csv")
    df.fillna("", inplace=True)
    df['TOTAL_COST'] = df['AVG_COST_PER_DAY'] + df['FLIGHT_COST']
    df['combined_features'] = (
        df['COUNTRY'].astype(str) + " " +
        df['CITY'].astype(str) + " " +
        df['THEME'].astype(str) + " " +
        df['HIGHLIGHTS'].astype(str) + " " +
        df['PRECAUTION'].astype(str)
    )
    return df

# ----------------- Image Fetching -----------------
def fetch_place_images(place_name):
    PIXABAY_API_KEY = "50958560-4c04addfc42591891ef9be539"
    url = "https://pixabay.com/api/"
    params = {
        "key": PIXABAY_API_KEY,
        "q": place_name,
        "image_type": "photo",
        "per_page": 3
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return [hit["webformatURL"] for hit in response.json()["hits"]]
    return []

# ----------------- Geolocation & Weather -----------------
def get_coordinates_osm(location):
    url = f"https://nominatim.openstreetmap.org/search?q={location}&format=json"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data:
            lat = float(data[0]['lat'])
            lon = float(data[0]['lon'])
            return lat, lon
    return None, None

def get_local_time(lat, lon, utc_time_str):
    utc_time = datetime.strptime(utc_time_str, "%Y-%m-%dT%H:%M:%SZ")
    utc_time = utc_time.replace(tzinfo=pytz.utc)
    tf = TimezoneFinder()
    timezone_str = tf.timezone_at(lat=lat, lng=lon)
    if timezone_str:
        local_time = utc_time.astimezone(pytz.timezone(timezone_str))
        return local_time.strftime("%Y-%m-%d %H:%M:%S"), timezone_str
    else:
        return utc_time_str, "UTC"

def get_metno_weather(lat, lon):
    headers = {"User-Agent": "weather-checker/1.0 contact@example.com"}
    url = f"https://api.met.no/weatherapi/locationforecast/2.0/compact?lat={lat}&lon={lon}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        timeslot = data["properties"]["timeseries"][0]
        timestamp_utc = timeslot["time"]
        details = timeslot["data"]["instant"]["details"]
        temperature = details.get("air_temperature", "N/A")
        humidity = details.get("relative_humidity", "N/A")
        wind_speed = details.get("wind_speed", "N/A")
        condition = "N/A"
        next_data = timeslot["data"].get("next_1_hours") or timeslot["data"].get("next_6_hours")
        if next_data and "summary" in next_data:
            condition = next_data["summary"].get("symbol_code", "N/A").replace("_", " ").capitalize()
        local_time, timezone = get_local_time(lat, lon, timestamp_utc)
        return {
            "local_time": local_time,
            "timezone": timezone,
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "condition": condition
        }
    else:
        return None

# ----------------- Quiz & Theme Handling -----------------
def get_quiz_result(df, theme):
    filtered = df[df["THEME"].str.lower() == theme.lower()]
    if filtered.empty:
        return df.sample(1).iloc[0]
    return filtered.sample(1).iloc[0]

def get_user_theme_weights():
    return {
        "culture": st.slider("Importance of culture", 0, 10, 5),
        "food": st.slider("Importance of food", 0, 10, 5),
        "nature": st.slider("Importance of nature", 0, 10, 5),
        "adventure": st.slider("Importance of adventure", 0, 10, 5)
    }

# ----------------- Display Functions -----------------
def plot_quiz_scores(scores):
    # Clamp negative values to zero
    categories = list(scores.keys())
    raw_values = list(scores.values())
    safe_values = [max(0.0, v) for v in raw_values]
    values = safe_values + [safe_values[0]]  # close the radar loop

    angles = [n / float(len(categories)) * 2 * 3.1416 for n in range(len(categories))] + [0]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    ax.fill(angles, values, 'b', alpha=0.3)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    # Optional: Set radial limits explicitly to 0–1
    ax.set_ylim(0, 3)

    st.pyplot(fig)


def display_funny_theme_label(matched_theme):
    funny_theme_labels = {
        "Culture": "Cultural Connoisseur 🏯",
        "Modern": "Urban Adventurer 🏙️",
        "Adventurous": "Thrill Seeker 🧗‍♀️",
        "Photography": "Scenic Explorer 📸"
    }
    label = funny_theme_labels.get(matched_theme, matched_theme)
    st.markdown(f"### 🧭 You are a :orange[{label}]!")

def display_weather(city, country):
    with st.spinner("⛅ Fetching current weather..."):
        lat, lon = get_coordinates_osm(f"{city}, {country}")
        if lat and lon:
            weather = get_metno_weather(lat, lon)
            if weather:
                st.info(
                    f"🕒 Local Time: {weather['local_time']} ({weather['timezone']})\n\n"
                    f"🌡 Temperature: {weather['temperature']}°C\n\n"
                    f"💧 Humidity: {weather['humidity']}%\n\n"
                    f"🌬 Wind: {weather['wind_speed']} m/s\n\n"
                    f"☁ Condition: {weather['condition']}"
                )
            else:
                st.warning("⚠ Weather data is unavailable.")
        else:
            st.warning("⚠ Could not find coordinates for this destination.")

def show_destination(row, include_weather=True, show_trip_days=True):
    col1, col2 = st.columns([1, 2])
    with col1:
        images = fetch_place_images(f"{row['CITY']}, {row['COUNTRY']}")
        if images:
            st.image(images, width=250)
    with col2:
        st.subheader(f"{row['CITY']}, {row['COUNTRY']}")
        st.markdown(f"**Theme**: {row['THEME']}")
        st.markdown(f"**Highlights**: {row['HIGHLIGHTS']}")
        st.markdown(f"**Flight Cost**: ${row['FLIGHT_COST']}")
        st.markdown(f"**Daily Cost**: ${row['AVG_COST_PER_DAY']}")
    if show_trip_days:
        days = st.number_input("🗓 How many days?", min_value=1, value=7, key=f"{row['CITY']}_days")
        total_cost = row['FLIGHT_COST'] + row['AVG_COST_PER_DAY'] * days
        st.success(f"💰 Estimated Total Cost for {int(days)} days: ${total_cost:.2f}")
    if include_weather:
        display_weather(row['CITY'], row['COUNTRY'])
    with st.expander("📄 Travel Essentials"):
        if row['VISA'].strip().lower() == "no":
            st.success("🟢 No visa required!")
        else:
            st.warning("🔴 Visa required!")
        st.warning(f"⚠ Caution: {row['PRECAUTION']}")


def show_multiple_destinations(results):
    for _, row in results.iterrows():
        show_destination(row, include_weather=False, show_trip_days=False)

# ----------------- Bookmarking -----------------
def add_to_bookmarks(match):
    favourite = {"CITY": match["CITY"], "COUNTRY": match["COUNTRY"]}
    if favourite not in st.session_state.favourites:
        st.session_state.favourites.append(favourite)
        st.success(f"✅ Added {match['CITY']} to your favourites!")
    else:
        st.info(f"📍 {match['CITY']} is already in your favourites.")

# ---------- Initialize Session State ----------
def init_state():
    defaults = {
        "page": "Home",
        "question_set_index": 0,
        "method": "🧠 Personality Test",
        "matched_destination": None,
        "top_category": None,
        "favourites": [],
        "chat_history": [],
        "chat_step": None,
        "theme_weights": {},
        "has_snowed": False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_state()


# ----------------- Chatbot -----------------
def chatbot_step_handler(user_msg, match):
    user_msg_lower = user_msg.lower()
    if st.session_state.chat_step == "liking_check":
        if "yes" in user_msg_lower:
            st.session_state.chat_step = "offer_save"
            return "That's great! Would you like to save this location to your 'Bookmark'?"
        elif "no" in user_msg_lower:
            st.session_state.chat_step = "offer_retake"
            return "That's too bad. Want to retake a different set of quiz questions?"
        else:
            return "Please respond with 'yes' or 'no'."
    elif st.session_state.chat_step == "offer_save":
        if "yes" in user_msg_lower:
            add_to_bookmarks(match)
            st.session_state.chat_step = "done"
            return f"✅ {match['CITY']} saved! Want to explore another country?"
        elif "no" in user_msg_lower:
            st.session_state.chat_step = "done"
            return "No problem! You can always check the sidebar for more suggestions."
        else:
            return "Please respond with 'yes' or 'no'."
    elif st.session_state.chat_step == "offer_retake":
        if "yes" in user_msg_lower:
            for key in list(st.session_state.keys()):
                if key.startswith("q") or key in ["matched_destination", "theme_scores", "top_category", "chat_step", "chat_history"]:
                    del st.session_state[key]
            st.session_state.question_set_index = random.randint(0, 2)
            st.rerun()
        elif "no" in user_msg_lower:
            st.session_state.chat_step = "custom_location"
            return "Do you have a specific country or city in mind?"
        else:
            return "Please respond with 'yes' or 'no'."
    elif st.session_state.chat_step == "custom_location":
        return handle_custom_location_query(user_msg)
    return "Let me know if you'd like to explore more options!"

def handle_custom_location_query(user_input):
    user_location = user_input.strip().title()
    df = st.session_state.get("data", None)
    if df is None:
        return "Sorry, I can't access the data right now."
    asian_countries = df["COUNTRY"].unique()
    asian_cities = df["CITY"].unique()
    matched_country = difflib.get_close_matches(user_location, asian_countries, n=1, cutoff=0.6)
    matched_city = difflib.get_close_matches(user_location, asian_cities, n=1, cutoff=0.6)
    if matched_city:
        city_data = df[df['CITY'] == matched_city[0]].iloc[0]
        return (
            f"📍 *{city_data['CITY']}, {city_data['COUNTRY']}*\n"
            f"- Highlights: {city_data['HIGHLIGHTS']}\n"
            f"- Flight Cost: ${city_data['FLIGHT_COST']}\n"
            f"- Avg Daily Cost: ${city_data['AVG_COST_PER_DAY']}\n"
            f"- Visa: {city_data['VISA']}\n"
            f"- Precaution: {city_data['PRECAUTION']}"
        )
    elif matched_country:
        country_data = df[df['COUNTRY'] == matched_country[0]]
        msg = f"Here's what I found for *{matched_country[0]}*:"
        for _, row in country_data.iterrows():
            msg += (
                f"\n📍 *{row['CITY']}*\n"
                f"- Highlights: {row['HIGHLIGHTS']}\n"
                f"- Flight Cost: ${row['FLIGHT_COST']}\n"
                f"- Avg Daily Cost: ${row['AVG_COST_PER_DAY']}\n"
                f"- Visa: {row['VISA']}\n"
                f"- Precaution: {row['PRECAUTION']}"
            )
        return msg
    else:
        return f"Sorry, I couldn't find info for '{user_location}'. Try another name."


TOP_10_COUNTRIES = [
    "Malaysia", "Thailand", "Vietnam", "South Korea", "Japan",
    "India", "Indonesia", "China", "Philippines", "Singapore"
]

PIXABAY_API_KEY = "50958560-4c04addfc42591891ef9be539"  # Replace with env variable in production

@st.cache_data
def load_top10_data():
    df = pd.read_csv("dataset_travel.csv")
    df.fillna("", inplace=True)
    if 'TOTAL_COST' not in df.columns:
        df["TOTAL_COST"] = df["FLIGHT_COST"] + df["AVG_COST_PER_DAY"]
    return df



def display_country_block(row):
    st.markdown(f"### 📌 {row['COUNTRY']}")
    with st.spinner("🖼 Fetching stunning images..."):
        images = fetch_place_images(f"{row['CITY']}, {row['COUNTRY']}")
        if images:
            cols = st.columns(3)
            for i in range(min(3, len(images))):
                with cols[i]:
                    st.image(images[i], use_container_width=True)
        else:
            st.warning("⚠ No images found for this location.")
    st.markdown(f"🎬 *Why You’ll Love It*: {row['HIGHLIGHTS']} 💖")
    st.markdown(f"✈ *Flight Cost*: ${row['FLIGHT_COST']}")
    st.markdown(f"🔥 *Daily Burn Rate*: ${row['AVG_COST_PER_DAY']}")
    st.markdown("---")

def render_top10_page():
    st.title("🌍 Top 10 Countries to Visit – Let's Gooo! 🎒")
    df = load_top10_data()

    for country in TOP_10_COUNTRIES:
        country_df = df[df['COUNTRY'].str.lower() == country.lower()]
        if not country_df.empty:
            row = country_df.sample(1).iloc[0]
            display_country_block(row)
        else:
            st.warning(f"No data found for {country}.")


def explain_dominant_style(theme_scores):
    """Explains the dominant style based on personality test scores."""

    # Categories and their descriptions
    styles_description = {
        "urban": "Urban style refers to a modern, city-inspired aesthetic. It typically includes sleek, contemporary designs with a focus on functionality and minimalism. This style often features elements like street art, industrial design, and modern architecture.",
        "culture": "Culture style is centered around the exploration of historical landmarks, art, traditions, and the richness of diverse cultures. It includes visiting museums, galleries, cultural festivals, and immersing oneself in the local customs.",
        "food": "Food style is all about culinary experiences. It includes exploring local delicacies, street food, gourmet restaurants, and food markets. This style is often about experiencing the taste of different cultures and cuisines.",
        "nature": "Nature style emphasizes a connection to the outdoors and natural landscapes. It celebrates the beauty of forests, mountains, beaches, and national parks. Ideal for those who seek peace and inspiration from nature.",
        "scenery": "Scenery style is focused on the visual beauty of the environment. It involves visiting places with breathtaking views like mountains, oceans, forests, or expansive plains. This style is ideal for those who love photography and nature.",
        "party": "Party style revolves around socializing, nightlife, and vibrant energy. It includes enjoying nightclubs, bars, concerts, and events that are filled with people, music, and excitement.",
        "adventure": "Adventure style appeals to those who seek thrill and excitement. It includes activities like hiking, mountain climbing, scuba diving, and other extreme sports. This style is for people who crave new experiences and challenges.",
        "relax": "Relax style is all about tranquility and peace. It focuses on activities like yoga, meditation, spa visits, and enjoying calm, serene environments like beaches or lakes, perfect for de-stressing and rejuvenating."
    }

    # Find the dominant style based on the highest score
    dominant_style = max(theme_scores, key=theme_scores.get)

    # Get the description of the dominant style
    explanation = styles_description.get(dominant_style, "No explanation available for this style.")

    # Return the result as an explanation
    return f"Your dominant personality style is *{dominant_style.capitalize()}*:\n\n{explanation}"


def explain_all_styles():
    """Styled HTML version with boxes for each travel style."""
    st.markdown("""
    <style>
    .boxed-style {
        background-color: #1e1e1e;
        border: 2px solid #444;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        color: white;
    }
    .boxed-style h3 {
        margin: 0 0 0.5rem 0;
    }
    .boxed-style p {
        margin: 0.3rem 0;
    }
    </style>

    ### ✨ Travel Personality Styles Explained

    <div class="boxed-style">
    <h3>🛸 Urban Explorer</h3>
    <p>🏙️ Modern, fast-paced, and city-inspired!</p>
    <p>Love skyscrapers, neon streets, stylish cafes, art districts, and buzzing nightlife? You thrive where the action never sleeps!</p>
    </div>

    <div class="boxed-style">
    <h3>🏯 Cultural Seeker</h3>
    <p>Traditions, heritage, and deep cultural dives!</p>
    <p>Temples, museums, historic streets, local art — you're all about soaking in centuries of stories and traditions.</p>
    </div>

    <div class="boxed-style">
    <h3>🍜 Culinary Adventurer</h3>
    <p>Flavors first, always!</p>
    <p>Street food, fusion dishes, spice markets — your travel route is guided by your tastebuds!</p>
    </div>

    <div class="boxed-style">
    <h3>🌿 Nature Wanderer</h3>
    <p>Wild and free!</p>
    <p>Mountains, forests, lakes — your soul finds peace and energy in the untouched outdoors.</p>
    </div>

    <div class="boxed-style">
    <h3>🌄 Scenery Chaser</h3>
    <p>Chasing the perfect view...</p>
    <p>Sunsets, coastlines, wide landscapes — your lens and heart are both focused on the breathtaking.</p>
    </div>

    <div class="boxed-style">
    <h3>💃 Party Goer</h3>
    <p>Dance, lights, and late nights!</p>
    <p>Music festivals, rooftop bars, clubs and beach parties — you're there for the vibe and the beat.</p>
    </div>

    <div class="boxed-style">
    <h3>🧗 Adventure Seeker</h3>
    <p>Thrill is your travel fuel!</p>
    <p>Diving, climbing, hiking, and daring new things — the bolder the better for you.</p>
    </div>

    <div class="boxed-style">
    <h3>🏖️ Relaxation Lover</h3>
    <p>Calm, peace, and total bliss.</p>
    <p>Spas, hammocks, beaches, and quiet escapes — your ideal day is about doing absolutely nothing stressful.</p>
    </div>

    <div style="font-size: 18px; margin-top: 1rem;">
    💡 <em>Each style influences your destination match. You can score high in multiple styles, but your dominant one helps shape the most personalized travel path for you.</em>
    </div>
    """, unsafe_allow_html=True)


def chatbot():
    match = st.session_state.get("matched_destination")

    if isinstance(match, pd.Series):
        match = match.to_dict()

    st.session_state.matched_destination = match

    st.markdown("## 💬 Chatbot Assistant")
    st.info("Please complete the personality quiz first")


    if "initialized_chatbot" not in st.session_state:
        st.session_state.chat_step = "liking_check"
        st.session_state.chat_history = [("assistant", "Are you satisfied with the country recommended?")]
        #st.session_state.favourites = []
        if "favourites" not in st.session_state:
            st.session_state.favourites = []
        st.session_state.data = load_data()
        if st.session_state.matched_destination is None:
            st.session_state.matched_destination = {
                "CITY": "Tokyo", "COUNTRY": "Japan", "HIGHLIGHTS": "Skytree, sushi",
                "FLIGHT_COST": 300, "AVG_COST_PER_DAY": 150,
                "VISA": "No", "PRECAUTION": "None", "THEME": "Modern"
            }
        st.session_state.initialized_chatbot = True

    for role, msg in st.session_state.chat_history:
        st.chat_message(role).write(msg)

    if user_input := st.chat_input("Type your message here..."):
        st.chat_message("user").write(user_input)
        st.session_state.chat_history.append(("user", user_input))
        user_msg_lower = user_input.strip().lower()
        match = st.session_state.matched_destination

        if st.session_state.chat_step == "liking_check":
            if "yes" in user_msg_lower:
                st.session_state.chat_step = "offer_save"
                msg = "That's great! Would you like to save this location to your 'Bookmark'?"
            elif "no" in user_msg_lower:
                st.session_state.chat_step = "offer_retake"
                msg = "That's too bad. Want to retake a different set of quiz questions?"
            else:
                msg = "Please respond with 'yes' or 'no'."

        elif st.session_state.chat_step == "offer_save":
            if "yes" in user_msg_lower:
                match = st.session_state.get("matched_destination")
                if isinstance(match, pd.Series):
                    match = match.to_dict()
                    st.session_state.matched_destination = match
                favourite = {"CITY": match["CITY"], "COUNTRY": match["COUNTRY"]}
                if "favourites" not in st.session_state:
                    st.session_state.favourites = []
                if favourite not in st.session_state.favourites:
                    st.session_state.favourites.append(favourite)
                    msg = f"✅ {match['CITY']} saved! Would you like to explore another country?"
                else:
                    msg = f"📍 {match['CITY']} is already in your favourites. Want to explore another country?"
                st.session_state.chat_step = "ask_to_explore_more"

            elif "no" in user_msg_lower:
                st.session_state.chat_step = "ask_to_explore_more"
                msg = "No problem! Would you like to explore another country?"
            else:
                msg = "Please respond with 'yes' or 'no'."

        elif st.session_state.chat_step == "ask_to_explore_more":
            if "yes" in user_msg_lower:
                st.session_state.chat_step = "custom_location"
                msg = "Great! Please tell me the country or city you're interested in."
            elif "no" in user_msg_lower:
                st.session_state.chat_step = "done"
                msg = "You're all set! 😊"
            else:
                msg = "Please respond with 'yes' or 'no'."

        elif st.session_state.chat_step == "offer_retake":
            if "yes" in user_msg_lower:
                for key in list(st.session_state.keys()):
                    if key.startswith("q") or key in [
                        "matched_destination", "theme_scores", "top_category",
                        "chat_step", "chat_history"
                    ]:
                        del st.session_state[key]
                st.session_state.question_set_index = random.randint(0, 2)
                st.rerun()
            elif "no" in user_msg_lower:
                st.session_state.chat_step = "custom_location"
                msg = "Do you have a specific country or city in mind?"
            else:
                msg = "Please respond with 'yes' or 'no'."

        elif st.session_state.chat_step == "custom_location":
            df = st.session_state.data
            user_location = user_input.strip().title()
            matched_country = df[df["COUNTRY"].str.lower() == user_location.lower()]
            matched_city = df[df["CITY"].str.lower() == user_location.lower()]
            if not matched_country.empty:
                row = matched_country.iloc[0]
            elif not matched_city.empty:
                row = matched_city.iloc[0]
            else:
                msg = f"❌ Sorry, I couldn't find info for '{user_location}'. Try another country or city."
                st.chat_message("assistant").write(msg)
                st.session_state.chat_history.append(("assistant", msg))
                return

            info = (
                f"📍 {row['CITY']}, {row['COUNTRY']}\n"
                f"- Theme: {row['THEME']}\n"
                f"- Highlights: {row['HIGHLIGHTS']}\n"
                f"- Flight Cost: ${row['FLIGHT_COST']}\n"
                f"- Avg Daily Cost: ${row['AVG_COST_PER_DAY']}\n"
                f"- Visa: {row['VISA']}\n"
                f"- Precaution: {row['PRECAUTION']}"
            )
            st.session_state.matched_destination = row
            st.session_state.chat_step = "follow_up_after_custom"
            st.chat_message("assistant").markdown(info)
            st.session_state.chat_history.append(("assistant", info))
            st.chat_message("assistant").write("Are you satisfied with this recommendation?")
            st.session_state.chat_history.append(("assistant", "Are you satisfied with this recommendation?"))
            return  # Prevent double message

        elif st.session_state.chat_step == "follow_up_after_custom":
            if "yes" in user_msg_lower:
                st.session_state.chat_step = "offer_save"
                msg = "Would you like me to bookmark this destination?"
            elif "no" in user_msg_lower:
                st.session_state.chat_step = "ask_to_explore_more"
                msg = "No problem! Would you like to explore another country?"
            else:
                msg = "Are you satisfied with this recommendation? (yes or no)"

        elif st.session_state.chat_step == "done":
            msg = "You're all set! 😊"

        else:
            msg = "Let me know if you'd like to explore more options!"

        st.chat_message("assistant").write(msg)
        st.session_state.chat_history.append(("assistant", msg))
        return

# ---------- Streamlit Page Setup ----------
st.set_page_config(page_title="Travel Personality App", page_icon="🌏", layout="wide")

st.markdown("""
    <style>
    div.stButton > button {
        background-color: transparent !important;
        color: white !important;
        font-size: 28px !important;
        font-family: 'Segoe UI', sans-serif !important;
        font-weight: 600 !important;
        text-align: left !important;
        width: 100% !important;
        padding: 0.8rem 1rem !important;
        border: none !important;
        border-radius: 10px !important;
        transition: background-color 0.3s ease, transform 0.2s ease;
        justify-content: flex-start !important;
        display: flex !important;
    }
    
    div.stButton > button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Style radio button labels to be white */
label[data-baseweb="radio"] > div {
    color: white !important;
}

/* Optional: increase label font size for better readability */
label[data-baseweb="radio"] {
    font-size: 16px !important;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_data():
    return load_data()

df = get_data()
if "page" not in st.session_state:
    st.session_state.page = "Home"

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background-color: #121212;
    color: white;
}
<style>
""", unsafe_allow_html=True)

# ---------- Particle Effect & Snow ----------
particles_js = """<script src='https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js'></script>
<div id='particles-js' style='position:fixed;width:100vw;height:100vh;z-index:-1;'></div>
<script>
particlesJS('particles-js', {
  "particles": {
    "number": {"value": 100},
    "color": {"value": "#ffffff"},
    "shape": {"type": "circle"},
    "opacity": {"value": 0.5},
    "size": {"value": 3, "random": true},
    "line_linked": {"enable": true, "distance": 120, "color": "#ffffff", "opacity": 0.4, "width": 1},
    "move": {"enable": true, "speed": 1}
  },
  "interactivity": {
    "events": {
      "onhover": {"enable": true, "mode": "grab"},
      "onclick": {"enable": true, "mode": "push"}
    },
    "modes": {"grab": {"distance": 140, "line_linked": {"opacity": 1}}, "push": {"particles_nb": 4}}
  },
  "retina_detect": true
});
</script>"""
components.html(particles_js, height=400, scrolling=False)

# Load local CSS
#def local_css(file_name):
#    with open(file_name) as f:
#        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


st.markdown("""
<style>
/* ❄️ Snowflake Styling - Fixed to not block sidebar toggle */
.snowflake {
  color: #FFF;
  font-size: 1em;
  font-family: Arial;
  text-shadow: 0 0 1px #000;
}

@keyframes snowflakes-fall {
  0% { top: -10% }
  100% { top: 100% }
}
@keyframes snowflakes-shake {
  0% { transform: translateX(0px) }
  50% { transform: translateX(40px) }
  100% { transform: translateX(0px) }
}

.snowflake {
  position: fixed;
  top: -10%;
  z-index: 10;  /* LOWER than the sidebar toggle */
  pointer-events: none; /* Let clicks pass through */
  animation-name: snowflakes-fall, snowflakes-shake;
  animation-duration: 10s, 3s;
  animation-timing-function: linear, ease-in-out;
  animation-iteration-count: infinite, infinite;
  animation-play-state: running, running;
}

/* Offset snowflakes to the middle/right only */
.snowflake:nth-of-type(0) { left: 20%; animation-delay: 0s, 0s; }
.snowflake:nth-of-type(1) { left: 30%; animation-delay: 1s, 1s; }
.snowflake:nth-of-type(2) { left: 40%; animation-delay: 2s, 0.5s; }
.snowflake:nth-of-type(3) { left: 50%; animation-delay: 3s, 2s; }
.snowflake:nth-of-type(4) { left: 60%; animation-delay: 4s, 1s; }
.snowflake:nth-of-type(5) { left: 70%; animation-delay: 5s, 1.5s; }
.snowflake:nth-of-type(6) { left: 80%; animation-delay: 6s, 2s; }
.snowflake:nth-of-type(7) { left: 90%; animation-delay: 7s, 2.5s; }
.snowflake:nth-of-type(8) { left: 75%; animation-delay: 3s, 1s; }
.snowflake:nth-of-type(9) { left: 85%; animation-delay: 2s, 1s; }
</style>
""", unsafe_allow_html=True)

animation_symbol = "❄"
st.markdown(
    f"""
        <div class="snowflake">{animation_symbol}</div>
        <div class="snowflake">{animation_symbol}</div>
        <div class="snowflake">{animation_symbol}</div>
        <div class="snowflake">{animation_symbol}</div>
        <div class="snowflake">{animation_symbol}</div>
        <div class="snowflake">{animation_symbol}</div>
        <div class="snowflake">{animation_symbol}</div>
        <div class="snowflake">{animation_symbol}</div>
        <div class="snowflake">{animation_symbol}</div>
        <div class="snowflake">{animation_symbol}</div>
        <div class="snowflake">{animation_symbol}</div>
        <div class="snowflake">{animation_symbol}</div>
        <div class="snowflake">{animation_symbol}</div>
        """,
    unsafe_allow_html=True,
)
st.session_state.has_snowed = True

# ---------- Sidebar ----------

with st.sidebar:
    st.markdown(
        """
        <style>
        .sidebar-title {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        .sidebar-title img {
            width: 60px;
            height: 60px;
            margin-right: 15px;
            border-radius: 10px;
        }
        .sidebar-title h1 {
            color: white;
            font-size: 30px;
            margin: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


    # Load and encode image as base64 for HTML embedding
    def get_base64_image(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()


    img_base64 = get_base64_image("logo.png")

    # Display image + title side by side
    st.sidebar.markdown(
        f"""
        <div class='sidebar-title'>
            <img src="data:image/png;base64,{img_base64}">
            <h1>Travel Tools</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("""
            <style>
            section[data-testid="stSidebar"] {
                background-color: #1e1e1e;
            }
            </style>
        """, unsafe_allow_html=True)


    if st.button("🏠 Home"):
        st.session_state.page = "Home"
    if st.button("🧠 Personality Quiz"):
        st.session_state.page = "Quiz"
    if st.button("🌍 Top 10 Countries"):
        st.session_state.page = "Top10"
    if st.button("🖥️ Chatbot"):
        st.session_state.page = "Chatbot"
    if st.button("ℹ️ Travel Styles"):
        st.session_state.page = "Styles"
    if st.button("💖 Bookmarks"):
        st.session_state.page = "Bookmarks"

# ---------- Page Router ----------
if st.session_state.page == "Home":
    st.title("🏠 Welcome to the Asia Travel Tool!")
    st.markdown("""
    Once you launch the app, you’ll be presented with a “home page”. From here, you can navigate to different sections based on what you want to do:

    ### 1. 🧠 *Personality Test (Personality Quiz)*
    This section lets you take a quiz to determine your travel style. You’ll answer questions about your preferences — like hiking vs. city tours or local food vs. international cuisine. 
    The quiz evaluates your preferences across categories like **Nature**, **Food**, **Adventure**, etc. After completing it, you'll get personalized destination recommendations.

    **Steps to Use:**
    - Go to the *“Personality Quiz”* section.
    - Answer each question by selecting from options like *Strongly Agree*, *Agree*, etc.
    - Submit the quiz to discover your dominant travel style (e.g., *"Adventurous"* or *"Culture Buff"*).
    - View a destination that best matches your personality.

    ---

    ### 2. 🌏 *Viewing Recommended Destinations*
    After finishing the quiz, recommended destinations will appear based on your answers. Each destination comes with flight cost, average daily cost, highlights, visa information, precautions, and weather updates. 
    The recommendations are generated by an intelligent **recommendation engine** using your inputs.

    **Steps to Use:**
    - Explore the listed destinations after the quiz.
    - Click *“Add to Bookmarks”* to save your favorite spots.

    ---

    ### 3. 🌍 *Top 10 Countries (Top 10 Countries)*
    Browse a curated list of 10 amazing countries to visit. This section shows the best cities in each, along with travel highlights and estimated costs.

    **Steps to Use:**
    - Go to the *“Top 10 Countries”* page.
    - Scroll through and explore detailed destination highlights.

    ---

    ### 4. 💖 *Viewing and Managing Bookmarks (Bookmarks)*
    Save and manage all your favorite destinations. If a place looks interesting, save it here for easy access later.

    **Steps to Use:**
    - Click *“Add to Bookmarks”* after viewing a recommendation.
    - Go to the *Bookmarks* section to see your saved places.
    - You can store multiple bookmarks and revisit them anytime.

    ---

    ### 5. 🖥 *Chatbot (Chatbot)*
    Let the chatbot guide your travel journey. It checks if you’re happy with your recommendation and lets you save it or retake the quiz.

    **Steps to Use:**
    - Visit the *Chatbot* section.
    - Respond to the bot prompts — *yes* to save, *no* to explore more.
    - The bot helps you refine your preferences and find better destinations.

    ---
    """)

elif st.session_state.page == "Quiz":
    st.header("🧠 What kind of traveler are you?")
    # Initialize needed keys
    if "question_set_index" not in st.session_state:
        st.session_state["question_set_index"] = 0
    if "method" not in st.session_state:
        st.session_state["method"] = "🧠 Personality Test"

    categories = ["scenery", "nature", "urban", "culture", "food", "party", "relax", "adventure"]
    category_map = {
        "scenery": "Photography", "nature": "Photography", "urban": "Modern",
        "culture": "Culture", "food": "Culture", "party": "Modern",
        "relax": "Adventurous", "adventure": "Adventurous"
    }
    funny_theme_labels = {
        "Photography": "🦜 Nature Nerd", "Modern": "🌆 City Hustler",
        "Culture": "🏯 Culture Buff", "Adventurous": "⛰ Adrenaline Junkie"
    }
    # ---------- Question Bank (Likert-based) ----------
    question_sets = [
        [  # Set 1
            ("I feel most alive when hiking through forests or nature trails.", "nature"),
            ("I enjoy the energy and pace of busy city streets.", "urban"),
            ("Exploring historical sites helps me connect with a place.", "culture"),
            ("I make it a point to try street food whenever I travel.", "food"),
            ("Nightlife is an essential part of my travel experiences.", "party"),
            ("I would rather attend a yoga retreat than go hiking.", "relax"),
            ("I seek out places that offer adventure activities like rafting or rock climbing.", "adventure"),
            ("Watching sunsets or sunrises makes my trip feel special.", "scenery"),
            ("Visiting temples and ruins fascinates me.", "culture"),
            ("Traditional markets are among my favorite places to visit.", "food"),
            ("I enjoy spending time in shopping malls and high-tech areas.", "urban"),
            ("I often look for festivals or clubbing options while traveling.", "party"),
            ("I find peace in national parks and scenic landscapes.", "nature"),
            ("Spa days are a must on my vacation.", "relax"),
            ("I love road trips with great scenic views along the way.", "scenery"),
            ("Museums and art galleries are part of every good trip.", "culture"),
            ("I enjoy tasting spicy or unfamiliar dishes.", "food"),
            ("Skyscrapers impress me more than rural views.", "urban"),
            ("I would love to try skydiving or bungee jumping.", "adventure"),
            ("Relaxing by the sea or a quiet lake is ideal for me.", "relax"),
            ("I go out of my way to visit places known for natural beauty.", "scenery"),
            ("I try regional dishes even if I don't know what's in them.", "food"),
            ("A trip isn’t complete without some nightlife fun.", "party"),
            ("I plan getaways to escape into peaceful nature.", "nature")
        ],
        [  # Set 2
            ("I would love staying in a treehouse or eco-lodge.", "nature"),
            ("Bright city lights excite me more than natural landscapes.", "urban"),
            ("I actively seek historical tours or museums on trips.", "culture"),
            ("I enjoy trying unusual snacks at local stalls.", "food"),
            ("Concerts or music venues are key attractions for me.", "party"),
            ("I would prefer a silent retreat over a city tour.", "relax"),
            ("I choose destinations based on the adventure activities available.", "adventure"),
            ("I travel to experience breathtaking landscapes.", "scenery"),
            ("I’m fascinated by ancient temples and religious architecture.", "culture"),
            ("I like to join food tours or cooking classes.", "food"),
            ("I prefer visiting cities with skyscrapers than countryside areas.", "urban"),
            ("I plan my trips around events like music festivals.", "party"),
            ("Hiking to waterfalls or forest spots excites me.", "nature"),
            ("Relaxation activities like massage or spa time are important to me.", "relax"),
            ("I visit destinations known for their picture-perfect views.", "scenery"),
            ("Driving through rugged terrains excites me.", "adventure"),
            ("I visit UNESCO sites whenever possible.", "culture"),
            ("Sampling traditional dishes is a big part of my travel.", "food"),
            ("I feel inspired by modern architecture and tech cities.", "urban"),
            ("A beach sunset is my favorite way to end the day.", "scenery"),
            ("Wildlife safaris or jungle treks are on my bucket list.", "nature"),
            ("I enjoy socializing in vibrant nightlife settings.", "party"),
            ("I prioritize well-being and mindfulness on trips.", "relax"),
            ("My ideal vacation involves physical activity and excitement.", "adventure")
        ],
        [  # Set 3
            ("I dream of staying in bamboo cabins or mountain lodges.", "nature"),
            ("I am drawn to futuristic cities and innovation hubs.", "urban"),
            ("I find temples and heritage sites more interesting than malls.", "culture"),
            ("I prefer local breakfasts to hotel buffets.", "food"),
            ("I choose destinations known for their parties and social scenes.", "party"),
            ("I enjoy quiet meditation in natural settings.", "relax"),
            ("Remote treks and difficult hikes appeal to me.", "adventure"),
            ("I travel to witness stunning scenery like mountains or coastlines.", "scenery"),
            ("Cities with a long history attract me.", "culture"),
            ("I want to try every local dish when traveling.", "food"),
            ("I love using public transport in large cities.", "urban"),
            ("Dancing at beach parties and night events is fun for me.", "party"),
            ("Natural springs or mountain streams are ideal places to swim.", "nature"),
            ("I spend part of my vacation reading or journaling.", "relax"),
            ("I visit places just to capture breathtaking views with my camera.", "scenery"),
            ("Mountain trekking or dune rallying sound exciting to me.", "adventure"),
            ("I prioritize cultural experiences over modern conveniences.", "culture"),
            ("Trying new and exotic flavors is one of my top priorities.", "food"),
            ("Rooftop views of big cities excite me.", "urban"),
            ("Nature helps me reflect and reconnect with myself.", "relax"),
            ("Observing animals in their natural environment excites me.", "nature"),
            ("I enjoy adrenaline-pumping sports while traveling.", "adventure"),
            ("A trip to beautiful landscapes leaves a lasting impression on me.", "scenery"),
            ("A night out with music makes the trip worthwhile.", "party")
        ]
    ]

    questions = question_sets[st.session_state["question_set_index"]]

    if st.session_state["method"] == "🧠 Personality Test":
        st.title("🌏 Asia Travel Personality Chatbot – Find Your Inner Explorer 🤸‍♂🍜🙺")
    likert = {"Strongly Disagree": 0.0,
              "Disagree": 0.25,
              "Neutral": 0.5,
              "Agree": 0.75,
              "Strongly Agree": 1.0}
    scores = {cat: 0.0 for cat in categories}

    with st.form("quiz_form"):
        unanswered = False  # Track if any question is skipped

        for idx, (q, cat) in enumerate(questions):
            st.markdown(f"<div style='font-size:20px; font-weight:bold;'>{idx + 1}. {q}</div>", unsafe_allow_html=True)
            ans = st.radio("", list(likert.keys()), key=f"quiz_q{idx}", index=None)
            if ans is None:
                unanswered = True
            else:
                scores[cat] += likert[ans]

        if st.form_submit_button("🎯 Get My Result"):
            if unanswered:
                st.warning("🚨 Please answer all questions before submitting!")
            else:
                top_cat = max(scores, key=scores.get)
                theme = category_map[top_cat]
                result = get_quiz_result(df, theme)
                st.session_state["matched_destination"] = result
                st.session_state["top_category"] = top_cat
                st.session_state["matched_theme"] = theme

    # --- Display destination OUTSIDE the form, so it reacts to widget changes ---
    match_theme = st.session_state.get("matched_theme")
    matched = st.session_state.get("matched_destination")
    if matched is not None and isinstance(matched, pd.Series):
        display_funny_theme_label(match_theme)
        show_destination(matched)
        plot_quiz_scores(scores)
        # Show personality explanation
        explanation = explain_dominant_style(scores)
        st.markdown("#### 🧬 Personality Insight")
        st.markdown(explanation)
        if st.button("💖 Add this to My Bookmark"):
            favourite = {"CITY": matched["CITY"], "COUNTRY": matched["COUNTRY"]}
            if "favourites" not in st.session_state:
                st.session_state.favourites = []
            if favourite not in st.session_state.favourites:
                st.session_state.favourites.append(favourite)
                st.success(f"✅ Added {matched['CITY']} to your favourites!")
            else:
                st.info(f"📍 {matched['CITY']} is already in your favourites.")
        # 🔁 Retake Quiz Button
        if st.button("🔁 Retake Quiz"):
            # Clear previous quiz-related state
            for key in list(st.session_state.keys()):
                if key.startswith("quiz_q") or key in [
                    "matched_destination", "top_category", "matched_theme", "theme_scores"
                ]:
                    del st.session_state[key]

            # Ensure a different question set
            current_index = st.session_state.get("question_set_index", 0)
            new_index = random.choice([i for i in range(3) if i != current_index])
            st.session_state.question_set_index = new_index

            st.rerun()

elif st.session_state.page == "Top10":
    render_top10_page()

elif st.session_state.page == "Bookmarks":
    st.title("💖 Bookmarks")
    if st.session_state.get("favourites"):
        for fav in st.session_state["favourites"]:
            st.markdown(f"📍 {fav['CITY']}, {fav['COUNTRY']}")
    else:
        st.info("No bookmarks yet.")


elif st.session_state.page == "Styles":
    st.title("🧳 Travel Personality Styles Explained")
    st.markdown(explain_all_styles(), unsafe_allow_html=True)

if st.session_state.page == "Chatbot":
    chatbot()




