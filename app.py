import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from urllib.parse import quote_plus

st.set_page_config(
    page_title='Tourist Destination Recommendation System',
    layout='wide'
)


@st.cache_data
def load_data(path="Indian_Tourism_Dataset_1000.csv"):
    df_local = pd.read_csv(path)

    # Convert any numeric-like columns to numeric and fill NaNs in numeric columns with the column mean
    numeric_cols = df_local.columns[df_local.dtypes.apply(lambda d: pd.api.types.is_numeric_dtype(d))].tolist()
    # Also attempt to coerce object columns that look numeric
    object_like_numeric = [c for c in df_local.columns if df_local[c].dtype == object and df_local[c].str.replace(',','').str.replace('.','').str.isnumeric().any()]
    for c in object_like_numeric:
        df_local[c] = pd.to_numeric(df_local[c].str.replace(',',''), errors='coerce')

    if numeric_cols:
        # Ensure numeric columns are numeric
        df_local[numeric_cols] = df_local[numeric_cols].apply(pd.to_numeric, errors='coerce')
        # Compute column means and round to 2 decimal places before filling
        means = df_local[numeric_cols].mean().round(2)
        df_local[numeric_cols] = df_local[numeric_cols].fillna(means)
    # Recompute numeric_cols in case coercion added more
    numeric_cols = df_local.select_dtypes(include=['number']).columns.tolist()
    # As a final safety, fill any remaining numeric NaNs with column mean rounded to 2 decimals
    if numeric_cols:
        df_local[numeric_cols] = df_local[numeric_cols].fillna(df_local[numeric_cols].mean().round(2))

    return df_local


@st.cache_data
def load_dataset_records(path="Indian_Tourism_Dataset_1000.csv"):
    return pd.read_csv(path)


# Load dataset (cached)
df = load_data()


def google_maps_url(destination):
    location = []
    seen = set()
    for field in ('Destination_Name', 'City', 'State'):
        value = destination.get(field)
        if pd.notna(value) and str(value).strip():
            value = str(value).strip()
            if value.casefold() not in seen:
                location.append(value)
                seen.add(value.casefold())
    query = ", ".join(location + ["India"])
    return f"https://www.google.com/maps/search/?api=1&query={quote_plus(query)}"


# --- Visualization helpers (Plotly) ---
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# Dashboard styling
st.markdown('''
<style>
    :root {
        color-scheme: light;
    }
    .stApp {
        --primary-color: #3b82a0;
        --background-color: #f8fafc;
        --secondary-background-color: #f1f4f7;
        --text-color: #1f2937;
        background: #f8fafc;
        color: #1f2937;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    [data-testid="stHeader"] {
        background: rgba(248, 250, 252, 0.92);
    }
    [data-testid="stSidebar"] {
        background: #f1f4f7;
        border-right: 1px solid #e3e8ee;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #536273;
    }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] label {
        color: #263445 !important;
        font-weight: 600;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [role="group"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5df !important;
        border-radius: 0.55rem;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] input {
        background-color: transparent !important;
        color: #1f2937 !important;
        -webkit-text-fill-color: #1f2937 !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] button {
        background-color: #ffffff !important;
        color: #526273 !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] svg {
        fill: currentColor;
    }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea {
        color: #1f2937 !important;
        -webkit-text-fill-color: #1f2937;
    }
    [role="listbox"] {
        background-color: #ffffff !important;
        color: #1f2937 !important;
    }
    [role="option"] {
        background-color: #ffffff !important;
        color: #1f2937 !important;
    }
    [role="option"]:hover,
    [role="option"][aria-selected="true"] {
        background-color: #eaf2f7 !important;
        color: #172536 !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 0.2rem;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        border-radius: 0.55rem;
        padding: 0.3rem 0.55rem;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: #e3edf4;
        color: #17445b;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked) > div > div > div:first-child {
        border-color: #3b82a0;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked) > div > div > div:first-child > div {
        background: #3b82a0;
    }
    .block-container {
        max-width: 1280px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3 {
        color: #172536;
        letter-spacing: -0.025em;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e7ecf1;
        border-radius: 0.8rem;
        padding: 1rem 1.1rem;
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div {
        color: #172536 !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border: 1px solid #cbd5df !important;
        border-radius: 0.85rem;
        box-shadow: 0 1px 2px rgba(23, 37, 54, 0.04);
    }
    div[class*="st-key-recommender-destination-card-"],
    div[class*="st-key-search-destination-card"],
    div[class*="st-key-home-exploration-card-"] {
        box-sizing: border-box;
        width: 100%;
        padding: 0.9rem 1rem;
        margin-bottom: 0.85rem;
        background: #ffffff;
        border: 2px solid #8da5b5 !important;
        border-radius: 0.85rem;
        box-shadow: 0 2px 5px rgba(23, 37, 54, 0.08);
    }
    div.stButton > button {
        background-color: #ffffff;
        border: 1px solid #cbd5df;
        color: #1f2937 !important;
        border-radius: 0.55rem;
        font-weight: 600;
    }
    div.stButton > button:hover {
        background-color: #eaf2f7;
        border-color: #9bb8c9;
        color: #172536 !important;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #e7ecf1;
        border-radius: 0.7rem;
        overflow: hidden;
    }
</style>
''', unsafe_allow_html=True)

# Geocoding removed per user request — coordinates and mapping are no longer displayed or auto-resolved.
# The project retains the dataset's raw lat/lon columns if present, but the app will not attempt
# to geocode or show locations. If needed in the future, a separate geocoding pipeline can be
# added to precompute corrected coordinates and store them in the CSV.


def rating_comparison(df, selected_name):
    # Show rating comparison focused on the selected destination's category (fallback to overall top)
    sel = df[df['Destination_Name'].str.lower() == str(selected_name).lower()]
    if not sel.empty and 'Category' in df.columns and pd.notna(sel.iloc[0].get('Category')):
        cat = sel.iloc[0]['Category']
        subset = df[df['Category'] == cat].sort_values('Rating', ascending=False).head(30)
        title = f'Rating Comparison - Category: {cat} (Top in Category)'
        if subset.empty:
            subset = df.sort_values('Rating', ascending=False).head(30)
            title = 'Rating Comparison (Top 30 Overall)'
    else:
        subset = df.sort_values('Rating', ascending=False).head(30)
        title = 'Rating Comparison (Top 30 Overall)'

    subset = subset[::-1]
    fig = px.bar(subset, x='Rating', y='Destination_Name', orientation='h', title=title,
                labels={'Rating':'Rating','Destination_Name':'Destination'},
                color_discrete_sequence=px.colors.sequential.Viridis)
    fig.update_layout(title={'font':{'size':15}}, xaxis={'title':{'font':{'size':12}}, 'tickfont':{'size':11}}, yaxis={'tickfont':{'size':11}})
    return fig


def budget_analysis(df, selected_name, selected_row):
    # Show budget vs rating for destinations related to the selected one (same category or same state)
    figs = []
    rel = df.copy()
    if pd.notna(selected_row.get('Destination_Name')):
        mask = pd.Series([True]*len(df))
        if 'Category' in df.columns and pd.notna(selected_row.get('Category')):
            mask = mask & (df['Category'] == selected_row['Category'])
        if 'State' in df.columns and pd.notna(selected_row.get('State')):
            # include same state as well
            mask = mask | (df['State'] == selected_row['State'])
        rel = df[mask]
        if rel.empty:
            rel = df.copy()

    if 'Budget_Per_Person_INR' in df.columns:
        scatter = px.scatter(rel, x='Budget_Per_Person_INR', y='Rating', color='Budget_Category', hover_name='Destination_Name',
                            title='Budget Per Person vs Rating (Related Destinations)', height=400, color_discrete_sequence=px.colors.sequential.Viridis)
        scatter.update_layout(title={'font':{'size':15}}, xaxis={'title':{'font':{'size':12}}, 'tickfont':{'size':11}}, yaxis={'title':{'font':{'size':12}}, 'tickfont':{'size':11}})
        # add selected point as a larger marker with label (keep color scheme intact)
        try:
            if pd.notna(selected_row.get('Budget_Per_Person_INR')):
                scatter.add_trace(go.Scatter(x=[selected_row['Budget_Per_Person_INR']], y=[selected_row['Rating']], mode='markers+text', marker=dict(size=14), text=[selected_row['Destination_Name']], textposition='top center', showlegend=False))
        except Exception:
            pass
        figs.append(scatter)

    # Budget category distribution within related set
    if 'Budget_Category' in df.columns:
        bc = rel['Budget_Category'].value_counts().reset_index()
        bc.columns = ['Budget_Category','Count']
        bar = px.bar(bc, x='Budget_Category', y='Count', title='Budget Category Distribution (Related Destinations)', height=350, color='Budget_Category',
                    color_discrete_sequence=px.colors.sequential.Viridis)
        bar.update_layout(title={'font':{'size':15}}, xaxis={'title':{'font':{'size':12}}, 'tickfont':{'size':11}}, yaxis={'title':{'font':{'size':12}}, 'tickfont':{'size':11}})
        figs.append(bar)
    return figs


def category_distribution(df, selected_category):
    # Bar chart of category distribution but show counts for categories and mark selected by ordering
    counts = df['Category'].value_counts().reset_index()
    counts.columns = ['Category','Count']
    # Move selected category to front for emphasis if present
    if selected_category in counts['Category'].values:
        sel_row = counts[counts['Category'] == selected_category]
        others = counts[counts['Category'] != selected_category]
        counts = pd.concat([sel_row, others], ignore_index=True)
    fig = px.bar(counts, x='Category', y='Count', title='Distribution of Category', height=400, color='Category',
                color_discrete_sequence=px.colors.sequential.Viridis)
    fig.update_layout(title={'font':{'size':15}}, xaxis={'title':{'font':{'size':12}}, 'tickfont':{'size':11}, 'tickangle':-45}, yaxis={'title':{'font':{'size':12}}, 'tickfont':{'size':11}})
    return fig


def season_analysis(df, season):
    # Show destinations for the same season and prioritize same category/state if possible
    same_season = df[df['Best_Season'] == season]
    if same_season.empty:
        return None
    # Show top by rating within the season
    counts = same_season.sort_values('Rating', ascending=False).head(20)[['Destination_Name','Rating','Category']]
    fig = px.bar(counts[::-1], x='Rating', y='Destination_Name', orientation='h', title=f'Top Destinations for {season}', height=500, color_discrete_sequence=px.colors.sequential.Viridis)
    fig.update_layout(title={'font':{'size':15}}, xaxis={'title':{'font':{'size':12}}, 'tickfont':{'size':11}}, yaxis={'tickfont':{'size':11}})
    return fig


def state_tourism_analysis(df, state):
    # Average rating by state and top destinations within the selected state
    st_avg = df.groupby('State')['Rating'].mean().reset_index().sort_values('Rating', ascending=False)
    fig = px.bar(st_avg[::-1], x='Rating', y='State', orientation='h', title='Average Rating by State', height=500,
                color_discrete_sequence=px.colors.sequential.Viridis)
    fig.update_layout(title={'font':{'size':15}}, xaxis={'title':{'font':{'size':12}}, 'tickfont':{'size':11}}, yaxis={'tickfont':{'size':11}})

    # Additional chart: top destinations in the selected state
    top_state = df[df['State'] == state].sort_values('Rating', ascending=False).head(15)
    if not top_state.empty:
        fig2 = px.bar(top_state[::-1], x='Rating', y='Destination_Name', orientation='h', title=f'Top Destinations in {state}', height=400, color_discrete_sequence=px.colors.sequential.Viridis)
        fig2.update_layout(title={'font':{'size':15}}, xaxis={'title':{'font':{'size':12}}, 'tickfont':{'size':11}}, yaxis={'tickfont':{'size':11}})
    else:
        fig2 = None

    return fig, fig2


def suitable_for_analysis(df, suitable_for_str):
    # Compare overall suitable-for counts vs within the selected category (if possible)
    traveler_types = ['Family','Couple','Solo','Friends','Senior']
    overall_counts = {t: df['Suitable_For'].str.contains(t, case=False, na=False).sum() for t in traveler_types}
    overall_df = pd.DataFrame({'Traveler':list(overall_counts.keys()), 'Overall':list(overall_counts.values())})

    # If a specific category is mentioned in the passed string, try to compute category-level counts
    cat = None
    # Attempt to detect category name by checking if any category equals the selected string
    if 'Category' in df.columns and isinstance(suitable_for_str, str):
        # Not a reliable method to detect category from suitable_for_str; caller may pass the row's Suitable_For string
        pass

    # Fallback: show only overall counts in grouped format
    dfm = overall_df.melt(id_vars='Traveler', value_vars=['Overall'], var_name='Scope', value_name='Count')
    fig = px.bar(dfm, x='Traveler', y='Count', color='Scope', barmode='group', title='Suitable For Distribution (Overall)', height=350, color_discrete_sequence=px.colors.sequential.Viridis)
    fig.update_layout(title={'font':{'size':15}}, xaxis={'title':{'font':{'size':12}}, 'tickfont':{'size':11}}, yaxis={'title':{'font':{'size':12}}, 'tickfont':{'size':11}})
    return fig


def show_rating_across_states_plotly(df):
    """
    Plotly bar chart showing average Rating by State (horizontal for readability).
    Returns a Plotly figure.
    """
    if 'State' not in df.columns or 'Rating' not in df.columns:
        raise ValueError('DataFrame must contain State and Rating columns')

    st_avg = df.groupby('State')['Rating'].mean().reset_index().sort_values('Rating', ascending=True)
    fig = px.bar(st_avg, x='Rating', y='State', orientation='h', title='Average Rating by State', color='Rating', color_continuous_scale=px.colors.sequential.Viridis)
    fig.update_layout(height=600, yaxis={'dtick':1}, margin={'l':200})
    return fig



def parse_recommended_months(df):
    """
    Parse the Recommended_Months column and return a month_counts Series in calendar order.
    """
    if 'Recommended_Months' not in df.columns:
        # return zeros if column missing
        month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        return pd.Series([0]*12, index=month_order)

    month_order = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ]

    month_map = {
        'January':'Jan', 'February':'Feb', 'March':'Mar',
        'April':'Apr', 'May':'May', 'June':'Jun',
        'July':'Jul', 'August':'Aug', 'September':'Sep',
        'October':'Oct', 'November':'Nov', 'December':'Dec'
    }

    months = []

    for value in df['Recommended_Months'].dropna():
        value = str(value)
        for full, short in month_map.items():
            value = value.replace(full, short)
        value = value.replace(" to ", "-")
        value = value.replace(" ", "")

        if "-" in value:
            parts = value.split("-")
            if len(parts) == 2 and parts[0] in month_order and parts[1] in month_order:
                start, end = parts
                start_index = month_order.index(start)
                end_index = month_order.index(end)
                # handle wrap-around ranges defensively
                if start_index <= end_index:
                    months.extend(month_order[start_index:end_index+1])
                else:
                    months.extend(month_order[start_index:] + month_order[:end_index+1])
            else:
                months.extend([p for p in value.split(',') if p])
        elif "," in value:
            months.extend([p for p in value.split(',') if p])
        else:
            if value:
                months.append(value)

    month_counts = pd.Series(months).value_counts().reindex(month_order, fill_value=0)
    return month_counts


def show_recommended_months_plotly(df):
    month_counts = parse_recommended_months(df)
    month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    month_counts = month_counts.reindex(month_order, fill_value=0)
    fig = px.bar(x=month_counts.index, y=month_counts.values, labels={'x':'Month','y':'Number of Recommendations'}, title='Most Recommended Months for Tourist Destinations', color=month_counts.values, color_continuous_scale=px.colors.sequential.Viridis)
    fig.update_layout(height=400)
    return fig, month_counts


st.sidebar.markdown(
    '<div style="padding: 0.5rem 0 1.1rem;">'
    '<div style="font-size: 0.76rem; letter-spacing: .12em; text-transform: uppercase; color: #718096;">'
    'INDIA · TRAVEL</div><div style="font-size: 1.18rem; font-weight: 700; color: #172536; margin-top: .28rem;">'
    'Indian Tourism Explorer</div></div>',
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Explore",
    ["Home", "Recommender", "Analytics", "Dataset", "Models", "About", "Search", "Favorites"],
    index=0,
    format_func=lambda section: {
        "Home": "⌂  Home",
        "Recommender": "✦  Recommender",
        "Analytics": "▥  Analytics",
        "Dataset": "▤  Dataset",
        "Models": "◌  Models",
        "About": "ⓘ  About",
        "Search": "⌕  Search",
        "Favorites": "♡  Favorites"
    }[section]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Recommendation filters**")
budget_opt = df['Budget_Category'].dropna().unique().tolist() if 'Budget_Category' in df.columns else []
category_opt = df['Category'].dropna().unique().tolist() if 'Category' in df.columns else []
season_opt = df['Best_Season'].dropna().unique().tolist() if 'Best_Season' in df.columns else []

budget_choice = st.sidebar.selectbox('Budget category', ['Any'] + budget_opt, key='sidebar_budget')
category_choice = st.sidebar.selectbox('Destination category', ['Any'] + category_opt, key='sidebar_category')
season_choice = st.sidebar.selectbox('Best season', ['Any'] + season_opt, key='sidebar_season')
traveller_choice = st.sidebar.selectbox(
    'Travelling with',
    ['Any', 'Family', 'Couple', 'Solo', 'Friends', 'Senior Citizen'],
    key='sidebar_traveller'
)

# --- Session state initialization ---
if 'favorites' not in st.session_state:
    st.session_state['favorites'] = []

# --- Home page ---
if page == "Home":
    st.caption("INDIA · DESTINATION DISCOVERY")
    st.title("Explore Indian Destinations")
    st.write("Find places to visit, compare real destination details, and explore travel patterns across the dataset.")

    state_count = df['State'].nunique() if 'State' in df.columns else 0
    average_rating = df['Rating'].mean() if 'Rating' in df.columns else None
    metric_cols = st.columns(3)
    metric_cols[0].metric("Destinations", f"{len(df):,}")
    metric_cols[1].metric("States & territories", f"{state_count:,}")
    if average_rating is not None:
        metric_cols[2].metric("Average tourism rating", f"{average_rating:.2f}")

    st.subheader("Start exploring")
    col1, col2, col3 = st.columns(3)
    with col1, st.container(border=True, key="home-exploration-card-recommendations"):
        st.subheader("Recommendations")
        st.write("Filter destinations by budget, category, season, and traveller type. Results are ranked by their dataset rating.")
    with col2, st.container(border=True, key="home-exploration-card-analytics"):
        st.subheader("Tourism analytics")
        st.write("Explore state ratings, traveller suitability, destination ratings, budget, and recommended travel months.")
    with col3, st.container(border=True, key="home-exploration-card-dataset"):
        st.subheader("Destination data")
        st.write(
            "Browse all 1,000 destination records and their available travel attributes. "
            "Explore the original dataset and its numeric summary."
        )

# --- Search page ---
elif page == "Search":
    st.caption("DESTINATION LOOKUP")
    st.title("Search destinations")
    st.write("Search by destination name to view its available travel details.")

    # Use main dataset (no session uploads)
    data_source = df

    # Upload removed from Search page per request

    destination_search = st.text_input("Enter Destination Name", key='search_input')
    if st.button("Search Destination"):
        if not destination_search:
            st.warning("Please enter a destination name to search.")
        else:
            destination_data = data_source[
                data_source['Destination_Name'].str.contains(destination_search, case=False, na=False)
            ]
            if len(destination_data) > 0:
                st.success(f"Found {len(destination_data)} matching destination(s)")
                row = destination_data.iloc[0]
                # store selected destination name in session_state for analytics tab
                st.session_state['selected_destination_name'] = row['Destination_Name']

                with st.container(key="search-destination-card"):
                    st.subheader(f"{row['Destination_Name']}")
                    st.caption(" · ".join(str(value) for value in [row.get('City'), row.get('State')] if pd.notna(value)))
                    st.link_button("Open in Google Maps", google_maps_url(row))
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Tourism rating", f"{row.get('Rating','N/A')}")
                    with col2:
                        if 'Budget_Per_Person_INR' in data_source.columns and pd.notna(row.get('Budget_Per_Person_INR')):
                            st.metric("Budget per person", f"₹{row.get('Budget_Per_Person_INR'):,}")
                        else:
                            st.metric("Budget category", row.get('Budget_Category','N/A'))
                    with col3:
                        if 'Adventure_Level' in data_source.columns and pd.notna(row.get('Adventure_Level')):
                            st.metric("Adventure level", row.get('Adventure_Level'))
                    with col4:
                        if 'Safety_Score' in data_source.columns and pd.notna(row.get('Safety_Score')):
                            st.metric("Safety score", row.get('Safety_Score'))

                    st.markdown("**Destination details**")
                    detail_cols = st.columns(2)
                    with detail_cols[0]:
                        st.write("**Category** ·", row.get('Category', '—'))
                        st.write("**Best season** ·", row.get('Best_Season', '—'))
                        st.write("**Suitable for** ·", row.get('Suitable_For', '—'))
                    with detail_cols[1]:
                        if 'Food_Cost_Per_Day_INR' in data_source.columns and pd.notna(row.get('Food_Cost_Per_Day_INR')):
                            st.write("**Food cost per day** ·", f"₹{row.get('Food_Cost_Per_Day_INR'):,}")
                        if 'Shopping_Score' in data_source.columns and pd.notna(row.get('Shopping_Score')):
                            st.write("**Shopping score** ·", row.get('Shopping_Score'))
                    if pd.notna(row.get('Description')):
                        st.write(row.get('Description'))

                    if st.button('Save to Favorites', key='search_save_favorite'):
                        fav = st.session_state.get('favorites', [])
                        if row['Destination_Name'] not in fav:
                            fav.append(row['Destination_Name'])
                            st.session_state['favorites'] = fav
                            st.success('Saved to favorites')
                        else:
                            st.info('Already in favorites')

            else:
                st.warning("Destination not found. Try another name.")

# --- Analytics page ---
elif page == "Analytics":
    st.caption("DATA OVERVIEW")
    st.title("Tourism analytics")
    st.write("Patterns calculated directly from the destination dataset.")
    data_filtered = df.copy()

    metric_cols = st.columns(4)
    metric_cols[0].metric("Destinations", f"{len(data_filtered):,}")
    if 'State' in data_filtered.columns:
        metric_cols[1].metric("States & territories", f"{data_filtered['State'].nunique():,}")
    if 'Rating' in data_filtered.columns:
        metric_cols[2].metric("Average rating", f"{data_filtered['Rating'].mean():.2f}")
    if 'Safety_Score' in data_filtered.columns:
        metric_cols[3].metric("Average safety score", f"{data_filtered['Safety_Score'].mean():.2f}")

    if 'State' in data_filtered.columns and 'Rating' in data_filtered.columns:
        with st.container(border=True):
            st.subheader("Average tourism rating by state")
            fig_states = show_rating_across_states_plotly(data_filtered)
            fig_states.update_layout(
                template="plotly_white",
                height=540,
                margin=dict(l=10, r=10, t=25, b=10),
                xaxis_title="Average destination rating",
                yaxis_title="State"
            )
            fig_states.update_xaxes(showgrid=True, gridcolor="#e9eef3")
            st.plotly_chart(fig_states, width="stretch")

    first_col, second_col = st.columns(2)
    with first_col, st.container(border=True):
        st.subheader("Traveller suitability")
        fig_trav = suitable_for_analysis(data_filtered, '')
        fig_trav.update_layout(template="plotly_white", height=360, title=None, margin=dict(l=10, r=10, t=15, b=10))
        fig_trav.update_yaxes(title="Destination records", showgrid=True, gridcolor="#e9eef3")
        st.plotly_chart(fig_trav, width="stretch")

    with second_col, st.container(border=True):
        st.subheader("Destination rating distribution")
        if 'Rating' in data_filtered.columns:
            fig_rating = px.histogram(
                data_filtered,
                x='Rating',
                nbins=16,
                labels={'Rating': 'Destination rating', 'count': 'Number of destinations'},
                color_discrete_sequence=['#3b82a0']
            )
            fig_rating.update_layout(template="plotly_white", height=360, title=None, margin=dict(l=10, r=10, t=15, b=10))
            fig_rating.update_traces(marker_line_color="#1f2937", marker_line_width=1)
            fig_rating.update_yaxes(showgrid=True, gridcolor="#e9eef3")
            st.plotly_chart(fig_rating, width="stretch")
        else:
            st.info("Rating data is not available.")

    first_col, second_col = st.columns(2)
    with first_col, st.container(border=True):
        st.subheader("Budget per person and rating")
        if 'Budget_Per_Person_INR' in data_filtered.columns and 'Rating' in data_filtered.columns:
            fig_budget = px.scatter(
                data_filtered,
                x='Budget_Per_Person_INR',
                y='Rating',
                hover_name='Destination_Name',
                labels={'Budget_Per_Person_INR': 'Budget per person (INR)', 'Rating': 'Destination rating'},
                color_discrete_sequence=['#3b82a0']
            )
            fig_budget.update_layout(template="plotly_white", height=390, margin=dict(l=10, r=10, t=15, b=10))
            fig_budget.update_xaxes(showgrid=True, gridcolor="#e9eef3")
            fig_budget.update_yaxes(showgrid=True, gridcolor="#e9eef3")
            st.plotly_chart(fig_budget, width="stretch")
        else:
            st.info("Budget per person or rating data is not available.")

    with second_col, st.container(border=True):
        st.subheader("Recommended travel months")
        if 'Recommended_Months' in data_filtered.columns:
            fig_months, _ = show_recommended_months_plotly(data_filtered)
            fig_months.update_layout(
                template="plotly_white",
                height=390,
                title=None,
                margin=dict(l=10, r=10, t=15, b=10),
                xaxis_title="Month",
                yaxis_title="Number of destination records"
            )
            fig_months.update_yaxes(showgrid=True, gridcolor="#e9eef3")
            st.plotly_chart(fig_months, width="stretch")
        else:
            st.info("Recommended-month data is not available.")

# --- Recommender page ---
elif page == "Recommender":
    st.caption("PREFERENCE-BASED DISCOVERY")
    st.title("Find your next destination")
    st.write("Set your travel preferences to see matching destinations, ranked by their recorded tourism rating.")

    # Use main df (no session uploads)
    data_source = df

    st.info("Choose Budget category, Destination category, Best season, and Travelling with in the sidebar to refine these results.")

    result = data_source.copy()
    if budget_choice != 'Any' and 'Budget_Category' in result.columns:
        result = result[result['Budget_Category'] == budget_choice]
    if category_choice != 'Any' and 'Category' in result.columns:
        result = result[result['Category'] == category_choice]
    if season_choice != 'Any' and 'Best_Season' in result.columns:
        result = result[result['Best_Season'] == season_choice]
    if traveller_choice != 'Any' and 'Suitable_For' in result.columns:
        result = result[result['Suitable_For'].str.contains(traveller_choice, case=False, na=False)]

    result = result.sort_values(by='Rating', ascending=False)
    if 'Destination_Name' in result.columns:
        result = result.drop_duplicates(subset='Destination_Name', keep='first')
    result = result.head(20)

    if not result.empty:
        st.subheader("Recommended destinations")
        st.caption(f"{len(result)} unique destinations · ranked by tourism rating")
        # Option to add all displayed to favorites
        if st.button('Add all displayed to Favorites'):
            fav = st.session_state.get('favorites', [])
            added = 0
            for name in result['Destination_Name'].tolist():
                if name not in fav:
                    fav.append(name)
                    added += 1
            st.session_state['favorites'] = fav
            st.success(f'Added {added} destinations to favorites')

        for idx, r in result.iterrows():
            with st.container(key=f"recommender-destination-card-{idx}"):
                title_col, action_col = st.columns([4, 1])
                with title_col:
                    st.subheader(r['Destination_Name'])
                    place = " · ".join(str(value) for value in [r.get('City'), r.get('State')] if pd.notna(value))
                    if place:
                        st.caption(place)
                    st.link_button("Open in Google Maps", google_maps_url(r))
                with action_col:
                    if st.button("Save", key=f"fav_{idx}", width="stretch"):
                        fav = st.session_state.get('favorites', [])
                        if r['Destination_Name'] not in fav:
                            fav.append(r['Destination_Name'])
                            st.session_state['favorites'] = fav
                            st.success(f"Saved {r['Destination_Name']} to favorites")
                        else:
                            st.info('Already in favorites')

                details = st.columns(4)
                details[0].metric("Rating", f"{r.get('Rating', '—')}")
                details[1].write("**Budget**")
                details[1].write(r.get('Budget_Category', '—'))
                details[2].write("**Category**")
                details[2].write(r.get('Category', '—'))
                details[3].write("**Best season**")
                details[3].write(r.get('Best_Season', '—'))
                if pd.notna(r.get('Description')):
                    st.write(r.get('Description'))

        # Download / persist favorites options
        st.markdown('---')
        if st.button('Export Favorites as CSV'):
            fav = st.session_state.get('favorites', [])
            if fav:
                fav_df = df[df['Destination_Name'].isin(fav)]
                st.download_button('Download favorites CSV', fav_df.to_csv(index=False).encode('utf-8'), file_name='favorites.csv', mime='text/csv')
            else:
                st.info('No favorites to export')

        if st.button('Save Favorites to disk'):
            fav = st.session_state.get('favorites', [])
            try:
                with open('favorites.json','w',encoding='utf-8') as f:
                    json.dump(fav, f, ensure_ascii=False, indent=2)
                st.success('Favorites saved to favorites.json')
            except Exception as e:
                st.error(f'Could not save favorites: {e}')

    else:
        st.warning('No destinations match all of your selected sidebar preferences.')
        st.info('Try changing or clearing one or more filters to see matching destinations.')

# --- Dataset page ---
elif page == "Dataset":
    st.caption("SOURCE DATA")
    st.title("Destination dataset")
    records = load_dataset_records()
    st.write("Browse the source records. Column names and values are shown as stored in the CSV.")
    st.caption(f"{len(records):,} rows · {len(records.columns):,} columns")
    data_tab, summary_tab = st.tabs(["Destination records", "Numeric summary"])
    with data_tab:
        st.dataframe(records, width="stretch", hide_index=True)
    with summary_tab:
        numeric_records = records.select_dtypes(include='number')
        if numeric_records.empty:
            st.info("There are no numeric columns to summarize.")
        else:
            st.dataframe(numeric_records.describe().T, width="stretch")

# --- Models page ---
elif page == "Models":
    st.caption("RECOMMENDATION APPROACH")
    st.title("How recommendations work")
    st.write("The project contains two documented recommendation approaches. The live Streamlit page and notebook prototype are separate.")

    model_cols = st.columns(2)
    with model_cols[0], st.container(border=True):
        st.subheader("Live app · preference ranking")
        st.write("The Streamlit recommender applies the selected budget, category, season, and traveller-type filters to the destination data.")
        st.write("Matching destinations are sorted by the existing **Rating** column and the first 20 are displayed.")
        st.caption("This is the recommender currently used by the app. It does not calculate similarity scores.")
    with model_cols[1], st.container(border=True):
        st.subheader("Notebook · content similarity")
        st.write("The project notebook builds combined destination text features with `CountVectorizer`, then calculates cosine similarity to retrieve related destinations.")
        st.caption("This notebook prototype is not selectable in the Streamlit app. The notebook does not report evaluation metrics.")

# --- About page ---
elif page == "About":
    st.caption("ABOUT THE PROJECT")
    st.title("Indian tourism, made easier to explore")
    st.write(
        "The Tourist Destination Recommendation System is an educational data science project "
        "for exploring destinations across India and narrowing options using travel preferences."
    )
    about_cols = st.columns(2)
    with about_cols[0], st.container(border=True):
        st.subheader("Project")
        st.write("The application supports destination search, preference-based recommendations, dataset exploration, analytics, and a favorites list.")
        st.write("The dashboard reads destination information from the project’s Indian tourism CSV dataset.")
    with about_cols[1], st.container(border=True):
        st.subheader("Technology")
        st.write("Python · Streamlit · Pandas · NumPy · Plotly · scikit-learn")
        st.caption("The requirements also include Matplotlib and Seaborn for notebook analysis.")

# --- Favorites page ---
elif page == "Favorites":
    st.caption("SAVED DESTINATIONS")
    st.title("Favorites")
    favs = st.session_state.get('favorites', [])

    st.markdown("---")
    if favs:
        for name in favs:
            row = df[df['Destination_Name'] == name]
            with st.container(border=True):
                info_col, view_col, remove_col = st.columns([6, 1, 1])
                with info_col:
                    st.subheader(name)
                    if not row.empty:
                        saved_row = row.iloc[0]
                        place = " · ".join(
                            str(value)
                            for value in [saved_row.get('City'), saved_row.get('State')]
                            if pd.notna(value)
                        )
                        if place:
                            st.caption(place)
                        st.link_button("Open in Google Maps", google_maps_url(saved_row))
                if view_col.button('View', key=f'view_{name}'):
                    if not row.empty:
                        r = row.iloc[0]
                        st.write(f"### {r.get('Destination_Name','')}")
                        st.write('State:', r.get('State',''))
                        st.write('Rating:', r.get('Rating',''))
                        st.write('Category:', r.get('Category',''))
                        st.write('Budget Category:', r.get('Budget_Category',''))
                if remove_col.button('Remove', key=f'rem_{name}'):
                    favs = [favorite for favorite in favs if favorite != name]
                    st.session_state['favorites'] = favs
                    st.rerun()

        st.markdown('---')
        # Export favorites
        if st.button('Export Favorites as CSV'):
            fav_df = df[df['Destination_Name'].isin(favs)]
            st.download_button('Download favorites CSV', fav_df.to_csv(index=False).encode('utf-8'), file_name='favorites.csv', mime='text/csv')

        # Save/load from disk
        cols2 = st.columns(2)
        if cols2[0].button('Save Favorites to disk'):
            try:
                with open('favorites.json', 'w', encoding='utf-8') as f:
                    json.dump(favs, f, ensure_ascii=False, indent=2)
                st.success('Favorites saved to favorites.json')
            except Exception as e:
                st.error(f'Could not save favorites: {e}')
        if cols2[1].button('Load Favorites from disk'):
            try:
                if os.path.exists('favorites.json'):
                    with open('favorites.json', 'r', encoding='utf-8') as f:
                        loaded = json.load(f)
                    st.session_state['favorites'] = loaded
                    st.success('Favorites loaded from favorites.json')
                    st.experimental_rerun()
                else:
                    st.info('No favorites.json file found in project root.')
            except Exception as e:
                st.error(f'Could not load favorites: {e}')

        if st.button('Clear All Favorites'):
            st.session_state['favorites'] = []
            st.success('Cleared favorites')
            st.experimental_rerun()
    else:
        st.info('No favorites saved yet — add some from Search, Home or Recommender.')
