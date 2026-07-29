import streamlit as st
import pandas as pd
import numpy as np


df = pd.read_csv("Indian_Tourism_Dataset_1000.csv")

# Convert any numeric-like columns to numeric and fill NaNs in numeric columns with the column mean
numeric_cols = df.columns[df.dtypes.apply(lambda d: pd.api.types.is_numeric_dtype(d))].tolist()
# Also attempt to coerce object columns that look numeric
object_like_numeric = [c for c in df.columns if df[c].dtype == object and df[c].str.replace(',', '').str.replace('.', '').str.isnumeric().any()]
for c in object_like_numeric:
    df[c] = pd.to_numeric(df[c].str.replace(',',''), errors='coerce')

if numeric_cols:
    # Ensure numeric columns are numeric
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    # Compute column means and round to 2 decimal places before filling
    means = df[numeric_cols].mean().round(2)
    df[numeric_cols] = df[numeric_cols].fillna(means)
# Recompute numeric_cols in case coercion added more
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
# As a final safety, fill any remaining numeric NaNs with column mean rounded to 2 decimals
if numeric_cols:
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean().round(2))

# --- Visualization helpers (Plotly + Matplotlib/Seaborn) ---
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


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


def show_rating_across_states_seaborn(df):
    """
    Simple, clean seaborn bar chart showing average Rating by State.
    Returns a matplotlib Figure.
    """
    # Defensive: ensure required columns exist
    if 'State' not in df.columns or 'Rating' not in df.columns:
        raise ValueError('DataFrame must contain State and Rating columns')

    try:
        order = df.groupby('State')['Rating'].mean().sort_values(ascending=False).index
    except Exception:
        order = df['State'].unique()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.set_style('whitegrid')
    palette = ["#00b4d8", "#023e8a", "#0077b6", "#0096c7", "#0077b6"]

    # Use ci=None to disable confidence intervals for a clean bar chart
    sns.barplot(x='State', y='Rating', data=df, order=order, palette=palette, edgecolor='black', ax=ax, ci=None)

    ax.set_xlabel("State", fontsize=12)
    ax.set_ylabel("Average Rating", fontsize=12)
    ax.set_title("Tourism Ratings Across States", fontsize=15)
    ax.tick_params(axis='x', rotation=90)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


def show_recommended_months(df):
    """
    Parse the Recommended_Months column and return a seaborn bar chart Figure
    showing counts for each month in calendar order, plus the month_counts Series.
    """
    if 'Recommended_Months' not in df.columns:
        raise ValueError('DataFrame must contain Recommended_Months column')

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
                # if parsing failed, try splitting by comma
                months.extend([p for p in value.split(',') if p])
        elif "," in value:
            months.extend([p for p in value.split(',') if p])
        else:
            if value:
                months.append(value)

    month_counts = pd.Series(months).value_counts().reindex(month_order, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.set_style('whitegrid')
    sns.barplot(x=month_counts.index, y=month_counts.values, palette='viridis', ax=ax)
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Destinations')
    ax.set_title('Most Recommended Months for Tourist Destinations')
    ax.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    return fig, month_counts


st.set_page_config(
    page_title='Tourist Destination Recommendation System',
    layout='wide'
)

# Top-level tabs for navigation
tabs = st.tabs(["Home", "Search", "Analytics", "Recommendations"])

# --- Home tab ---
with tabs[0]:
    st.title(" Tourist Destination Recommendation System")
    st.write(
        """
        Discover the best tourist destinations in India based on your preferences.
        
        Explore destinations, check ratings, budget requirements, suitable travellers,
        seasons and get personalized recommendations.
        """
    )
    st.markdown("---")

    # Render a simple, clean seaborn chart below the Home content
    try:
        fig = show_rating_across_states_seaborn(df)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Could not render Ratings Across States chart: {e}")

    # Render the 'Most Recommended Months' chart below the ratings chart
    try:
        fig_months, month_counts = show_recommended_months(df)

        # Title above the months chart
        st.subheader("Most Recommended Months for Tourist Destinations")

        # Render figure
        st.pyplot(fig_months)

        # Compute and show a short insight summary under the chart
        total = month_counts.sum()
        if total > 0:
            top = month_counts.sort_values(ascending=False)
            top_nonzero = top[top > 0]
            if not top_nonzero.empty:
                top3 = top_nonzero.head(3)
                top_list = ', '.join([f"{m} ({v})" for m, v in top3.items()])
                peak = top_nonzero.idxmax()
                st.markdown(f"**Insight:** The dataset recommends destinations most often during **{peak}**. The top months are: {top_list}. In total {int(total)} month-recommendation entries were parsed.")
            else:
                st.info("No month recommendation entries were found in the data.")
        else:
            st.info("No month recommendation entries were found in the data.")

    except Exception as e:
        # Non-fatal: show a warning so the rest of the page still renders
        st.warning(f"Could not render Recommended Months chart: {e}")

# --- Search tab ---
with tabs[1]:
    st.header(" Explore Any Tourist Destination")
    destination_search = st.text_input("Enter Destination Name", key='search_input')
    if st.button("Search Destination"):
        destination_data = df[
            df['Destination_Name'].str.contains(destination_search, case=False, na=False)
        ]
        if len(destination_data) > 0:
            st.success("Destination Found")
            row = destination_data.iloc[0]
            # store selected destination name in session_state for analytics tab
            st.session_state['selected_destination_name'] = row['Destination_Name']

            st.subheader(f" {row['Destination_Name']}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tourism Rating", f"{row['Rating']} ")
            with col2:
                if 'Budget_Per_Person_INR' in df.columns:
                    st.metric("Budget Per Person", f"₹ {row['Budget_Per_Person_INR']}")
                else:
                    st.metric("Budget Category", row['Budget_Category'])
            with col3:
                if 'Adventure_Level' in df.columns:
                    st.metric("Adventure Level", row['Adventure_Level'])

            st.write("### Destination Information")
            col1, col2 = st.columns(2)
            with col1:
                st.write(" State:", row['State'])
                st.write("Category:", row['Category'])
                st.write(" Best Season:", row['Best_Season'])
            with col2:
                st.write(" Suitable For:", row['Suitable_For'])
                if 'Food_Cost_Per_Day_INR' in df.columns:
                    st.write(" Food Cost Per Day:", f"₹ {row['Food_Cost_Per_Day_INR']}")
                if 'Shopping_Score' in df.columns:
                    st.write(" Shopping Score:", row['Shopping_Score'])
        else:
            st.warning("Destination not found. Try another name.")

# --- Analytics tab ---
with tabs[2]:
    st.header("Visual Analytics")
    if 'selected_destination_name' not in st.session_state:
        st.info("No destination selected. Search for a destination in the 'Search' tab first.")
    else:
        sel_name = st.session_state['selected_destination_name']
        row = df[df['Destination_Name'].str.lower() == sel_name.lower()].iloc[0]

        st.subheader(f"Analytics for: {row['Destination_Name']}")
        st.markdown("---")
        # Simple, destination-focused visuals (no comparisons, not fancy)
        col1, col2 = st.columns(2)
        with col1:
            # Rating as a simple metric
            if 'Rating' in row.index and pd.notna(row.get('Rating')):
                st.metric('Rating', f"{row['Rating']}")
            else:
                st.info('No numeric rating available for this destination.')

            # Budget per person simple bar or category metric
            if 'Budget_Per_Person_INR' in df.columns and pd.notna(row.get('Budget_Per_Person_INR')):
                try:
                    bp = float(row.get('Budget_Per_Person_INR'))
                    st.write('Budget Per Person (INR)')
                    st.bar_chart(pd.DataFrame({'Budget':[bp]}, index=[row['Destination_Name']]))
                except Exception:
                    if 'Budget_Category' in row.index and pd.notna(row.get('Budget_Category')):
                        st.metric('Budget Category', row['Budget_Category'])
                    else:
                        st.info('No budget data available for this destination.')
            else:
                if 'Budget_Category' in row.index and pd.notna(row.get('Budget_Category')):
                    st.metric('Budget Category', row['Budget_Category'])
                else:
                    st.info('No budget data available for this destination.')

        with col2:
            # Category and related simple metrics
            if 'Category' in row.index and pd.notna(row.get('Category')):
                st.metric('Category', row['Category'])
            else:
                st.info('No category information available.')

            if 'Adventure_Level' in row.index and pd.notna(row.get('Adventure_Level')):
                st.metric('Adventure Level', row['Adventure_Level'])
            if 'Shopping_Score' in row.index and pd.notna(row.get('Shopping_Score')):
                st.metric('Shopping Score', row['Shopping_Score'])
            if 'Food_Cost_Per_Day_INR' in row.index and pd.notna(row.get('Food_Cost_Per_Day_INR')):
                st.metric('Food Cost Per Day (INR)', f"₹ {row['Food_Cost_Per_Day_INR']}")

        st.markdown('---')
        # Season and suitability (simple text/metrics)
        col3, col4 = st.columns(2)
        with col3:
            if 'Best_Season' in row.index and pd.notna(row.get('Best_Season')):
                st.metric('Best Season', row['Best_Season'])
            else:
                st.info('No season information available for this destination.')
        with col4:
            if 'Suitable_For' in row.index and pd.notna(row.get('Suitable_For')):
                st.write('Suitable For:')
                st.write(row['Suitable_For'])
            else:
                st.info('No suitability information available for this destination.')

        # Note: Removed map/location display as requested


# --- Recommendations tab ---
with tabs[3]:
    st.header(" Personalized Destination Recommendation")
    st.sidebar.header("Choose Your Preferences")

    budget = st.sidebar.selectbox(
        "Select Budget",
        df['Budget_Category'].unique()
    )

    category = st.sidebar.selectbox(
        "Select Destination Category",
        df['Category'].unique()
    )

    season = st.sidebar.selectbox(
        "Select Season",
        df['Best_Season'].unique()
    )

    traveller = st.sidebar.selectbox(
        "Travelling With",
        [
            "Family",
            "Couple",
            "Solo",
            "Friends",
            "Senior Citizen"
        ]
    )

    if st.sidebar.button("Recommend"):
        result = df[
            (df['Budget_Category'] == budget) &
            (df['Category'] == category) &
            (df['Best_Season'] == season) &
            (df['Suitable_For'].str.contains(traveller, case=False, na=False))
        ]
        result = result.sort_values(by='Rating', ascending=False).head(10)

        if len(result) > 0:
            st.subheader(" Top Recommended Destinations")
            for index,row in result.iterrows():
                st.write(f"### {row['Destination_Name']}")
                col1,col2 = st.columns(2)
                with col1:
                    st.write(" State:", row['State'])
                    st.write(" Rating:", row['Rating'])
                with col2:
                    st.write(" Budget:", row['Budget_Category'])
                    st.write(" Category:", row['Category'])
                st.divider()
        else:
            st.warning("No destination found. Try changing preferences.")