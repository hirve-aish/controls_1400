import streamlit as st

# -------------------------- Initialize session state keys --------------------------
if 'master_df' not in st.session_state:
    st.session_state['master_df'] = None         # Combined control data DataFrame

if 'audit_log' not in st.session_state:
    st.session_state['audit_log'] = []           # List for audit entries/change logs

if 'rules' not in st.session_state:
    st.session_state['rules'] = []                # Business rules list/dict

# You can add more persistent keys here if your app requires them
# For example: comments, flags, user info, etc.


# -------------------------- Step functions --------------------------

def step_1_upload_data():
    st.header("Step 1: Upload Data")
    from io import BytesIO
    import pandas as pd
    
    uploaded_files = st.file_uploader(
        "Upload one or more Excel files (one per control category):", 
        type=["xlsx"], accept_multiple_files=True
    )
    if uploaded_files:
        data_frames = []
        for f in uploaded_files:
            df = pd.read_excel(f)
            df['__source_file'] = f.name  # Track origin file name
            data_frames.append(df)
        master_df = pd.concat(data_frames, ignore_index=True)
        st.session_state['master_df'] = master_df
        st.success(f"Successfully loaded {len(master_df)} controls from {len(uploaded_files)} files.")
        st.dataframe(master_df.head(20))
    else:
        st.info("Please upload Excel files to begin.")


def step_2_data_integration():
    st.header("Step 2: Data Integration")
    import pandas as pd
    
    if st.session_state['master_df'] is None:
        st.warning("Please complete Step 1 to upload data first.")
        return
    
    df = st.session_state['master_df']
    
    # Example summary: count controls by source file
    source_summary = df['__source_file'].value_counts().reset_index()
    source_summary.columns = ['Source File', 'Control Count']

    st.subheader("Control Categories & Counts")
    st.table(source_summary)

    st.subheader("Preview of Integrated Controls Data")
    st.dataframe(df.head(20))


def step_3_interactive_exploration():
    st.header("Step 3: Interactive Data Exploration")
    import pandas as pd
    
    if st.session_state['master_df'] is None:
        st.warning("Please upload data in Step 1 before exploring.")
        return
    
    df = st.session_state['master_df']

    # Sidebar filters (example: Control Category, Owner, Flagged)
    st.sidebar.header("Filters")

    filtered_df = df.copy()

    if "Control Category" in df.columns:
        categories = df["Control Category"].dropna().unique()
        selected_cats = st.sidebar.multiselect("Control Category", categories, default=categories)
        filtered_df = filtered_df[filtered_df["Control Category"].isin(selected_cats)]

    if "Owner" in df.columns:
        owners = df["Owner"].dropna().unique()
        selected_owners = st.sidebar.multiselect("Owner", owners, default=owners)
        filtered_df = filtered_df[filtered_df["Owner"].isin(selected_owners)]

    if "Flagged" in df.columns:
        flagged_only = st.sidebar.checkbox("Show flagged only")
        if flagged_only:
            filtered_df = filtered_df[filtered_df["Flagged"] == True]

    st.success(f"Showing {len(filtered_df)} controls after filtering.")

    # Editable table
    edited_df = st.data_editor(filtered_df, key="data_editor", num_rows="dynamic", use_container_width=True)
    
    # Write edits back to master_df
    # Note: This simplified example replaces filtered rows,
    # in your code you may want to properly merge / update master_df rows by index or unique ID.
    st.session_state['master_df'].update(edited_df)
    
    st.write("Updated Data Preview:")
    st.dataframe(edited_df.head(20))


def step_4_real_time_collaboration():
    st.header("Step 4: Real-time Editing & Collaboration")
    import pandas as pd
    from datetime import datetime

    if st.session_state['master_df'] is None:
        st.warning("Please complete previous steps first.")
        return

    user = st.sidebar.text_input("Enter your username", value="User1")
    df = st.session_state['master_df']

    if "Flagged" not in df.columns:
        df["Flagged"] = False
    
    if "Comments" not in df.columns:
        df["Comments"] = ""

    if "audit_log" not in st.session_state:
        st.session_state["audit_log"] = []

    def log_change(row_idx, col, old, new):
        st.session_state["audit_log"].append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "User": user,
            "Row": row_idx,
            "Column": col,
            "Old Value": old,
            "New Value": new
        })

    edited_df = st.data_editor(df, key="collab_editor", num_rows="dynamic", use_container_width=True)

    # Detect changes and log them
    for i in edited_df.index:
        if not edited_df.loc[i].equals(df.loc[i]):
            for col in df.columns:
                old_val = df.at[i, col]
                new_val = edited_df.at[i, col]
                if old_val != new_val:
                    log_change(i, col, old_val, new_val)

    st.session_state['master_df'] = edited_df.copy()

    st.markdown("### Flag or Unflag Controls")
    selected_rows = st.multiselect("Select control rows to flag/unflag:", edited_df.index.tolist())

    if selected_rows:
        action = st.radio("Flag or Unflag?", ["Flag", "Unflag"], index=0)
        if st.button("Apply"):
            for idx in selected_rows:
                old_flag = st.session_state['master_df'].at[idx, "Flagged"]
                new_flag = (action == "Flag")
                if old_flag != new_flag:
                    st.session_state['master_df'].at[idx, "Flagged"] = new_flag
                    log_change(idx, "Flagged", old_flag, new_flag)
            st.success(f"{action}ged {len(selected_rows)} controls.")

    st.info(f"Currently flagged controls: {st.session_state['master_df']['Flagged'].sum()}")


def step_5_audit_trail_logging():
    st.header("Step 5: Audit Trail Logging")

    audit_log = st.session_state.get('audit_log', [])
    if not audit_log:
        st.info("No changes yet. Make edits to generate audit logs.")
        return

    import pandas as pd

    df = pd.DataFrame(audit_log)
    st.dataframe(df.sort_values("Timestamp", ascending=False), height=400)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Audit Log as CSV", data=csv, file_name="audit_log.csv", mime="text/csv")


def step_6_rules_management():
    st.header("Step 6: Rules Management & Dynamic Updates")
    import pandas as pd
    import numpy as np

    if st.session_state['master_df'] is None:
        st.warning("Please complete prior steps first.")
        return

    df = st.session_state['master_df']

    if "Flagged" not in df.columns:
        df["Flagged"] = False
    if "Risk Score" not in df.columns:
        df["Risk Score"] = np.random.randint(1, 10, size=len(df))

    # Initialize default rules if empty
    if not st.session_state['rules']:
        st.session_state['rules'] = [
            {
                "Rule Name": "High Risk Flagging",
                "Description": "Flag controls with Risk Score >= Threshold",
                "Threshold": 7,
                "Active": True
            },
            {
                "Rule Name": "Low Risk Unflagging",
                "Description": "Unflag controls with Risk Score < Threshold",
                "Threshold": 4,
                "Active": True
            }
        ]

    rules_df = pd.DataFrame(st.session_state['rules'])
    edited_rules = st.data_editor(rules_df, num_rows="dynamic", key="rules_editor")
    st.session_state['rules'] = edited_rules.to_dict('records')

    if st.button("Apply Rules"):
        new_df = df.copy()
        for rule in st.session_state['rules']:
            if not rule.get("Active", True):
                continue
            name = rule.get("Rule Name", "")
            threshold = rule.get("Threshold", None)
            if name == "High Risk Flagging" and threshold is not None:
                condition = new_df["Risk Score"] >= threshold
                new_df.loc[condition, "Flagged"] = True
            elif name == "Low Risk Unflagging" and threshold is not None:
                condition = new_df["Risk Score"] < threshold
                new_df.loc[condition, "Flagged"] = False
        st.session_state['master_df'] = new_df
        st.success("Rules applied.")


def step_7_advanced_visualization():
    st.header("Step 7: Advanced Slicing & Interactive Visualization")
    import plotly.express as px

    if st.session_state['master_df'] is None:
        st.warning("Please upload and prepare your data in earlier steps.")
        return

    df = st.session_state['master_df']

    # Sidebar filters
    st.sidebar.header("Filters")

    filtered_df = df.copy()

    if "Control Category" in df.columns:
        cats = df["Control Category"].dropna().unique()
        sel_cats = st.sidebar.multiselect("Control Category", cats, default=cats)
        filtered_df = filtered_df[filtered_df["Control Category"].isin(sel_cats)]

    if "Owner" in df.columns:
        owners = df["Owner"].dropna().unique()
        sel_owners = st.sidebar.multiselect("Owner", owners, default=owners)
        filtered_df = filtered_df[filtered_df["Owner"].isin(sel_owners)]

    if "Flagged" in df.columns:
        flagged_only = st.sidebar.checkbox("Flagged Only", value=False)
        if flagged_only:
            filtered_df = filtered_df[filtered_df["Flagged"] == True]

    st.write(f"Showing {len(filtered_df)} controls after filtering")

    chart_type = st.selectbox("Select Chart Type", ["Bar Chart", "Scatter Plot", "Pie Chart"])

    if chart_type == "Bar Chart":
        x_options = [col for col in filtered_df.columns if filtered_df[col].dtype == 'object']
        y_options = [col for col in filtered_df.columns if filtered_df[col].dtype in ['int64', 'float64']]
        if not x_options or not y_options:
            st.warning("Need categorical and numeric columns for Bar Chart.")
            return
        x_axis = st.selectbox("X Axis", x_options, index=0)
        y_axis = st.selectbox("Y Axis", y_options, index=0)
        bar_data = filtered_df.groupby(x_axis)[y_axis].mean().reset_index()
        fig = px.bar(bar_data, x=x_axis, y=y_axis, title=f"Average {y_axis} by {x_axis}")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Scatter Plot":
        numeric_cols = [col for col in filtered_df.columns if filtered_df[col].dtype in ['int64', 'float64']]
        if len(numeric_cols) < 2:
            st.warning("Need at least two numeric columns for Scatter Plot.")
            return
        x_axis = st.selectbox("X Axis", numeric_cols, index=0)
        y_axis = st.selectbox("Y Axis", numeric_cols, index=1)
        color_options = [col for col in filtered_df.columns if filtered_df[col].dtype == 'object'] + [None]
        color_by = st.selectbox("Color By", color_options, index=len(color_options)-1)
        fig = px.scatter(filtered_df, x=x_axis, y=y_axis, color=color_by if color_by else None,
                         title=f"Scatter: {y_axis} vs {x_axis}")
        st.plotly_chart(fig, use_container_width=True)

    else:  # Pie Chart
        cat_cols = [col for col in filtered_df.columns if filtered_df[col].dtype == 'object']
        if not cat_cols:
            st.warning("Need at least one categorical column for Pie Chart.")
            return
        pie_by = st.selectbox("Category for Pie Chart", cat_cols)
        pie_data = filtered_df[pie_by].value_counts().reset_index()
        pie_data.columns = [pie_by, "Count"]
        fig = px.pie(pie_data, names=pie_by, values="Count", title=f"Distribution by {pie_by}")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show Filtered Data"):
        st.dataframe(filtered_df.reset_index(drop=True))


def step_8_review_and_resolve_flags():
    st.header("Step 8: Review & Resolve Flagged Controls")
    if st.session_state['master_df'] is None:
        st.warning("Please complete prior steps first.")
        return

    df = st.session_state['master_df']

    if "Flagged" not in df.columns:
        df["Flagged"] = False
    
    if "Comments" not in df.columns:
        df["Comments"] = ""

    flagged_controls = df[df["Flagged"] == True]
    if flagged_controls.empty:
        st.info("No flagged controls to review.")
        return

    st.markdown(f"### {len(flagged_controls)} Flagged Controls Pending Review")

    with st.form("flag_review_form"):
        updates = []
        for idx, row in flagged_controls.iterrows():
            st.markdown(f"---\n**Control Index:** {idx}")
            st.write(row.drop(["Flagged", "Comments", "__source_file"], errors='ignore'))
            comment = st.text_area(f"Comment for control {idx}:", value=row["Comments"] or "", key=f"comment_{idx}")
            resolved = st.checkbox(f"Mark control {idx} as resolved", key=f"resolve_{idx}")
            updates.append((idx, comment, resolved))

        submitted = st.form_submit_button("Save Updates")
        if submitted:
            for idx, comment, resolved in updates:
                df.at[idx, "Comments"] = comment
                if resolved:
                    df.at[idx, "Flagged"] = False
            st.session_state['master_df'] = df
            st.success("Updated flagged controls.")

    st.subheader("Remaining Flagged Controls")
    st.dataframe(df[df["Flagged"] == True].reset_index(drop=True))


def step_9_export_data():
    st.header("Step 9: Export Updated Controls Data")
    import pandas as pd
    from io import BytesIO

    if st.session_state['master_df'] is None:
        st.warning("Please complete prior steps before exporting data.")
        return

    df = st.session_state['master_df']

    if "__source_file" not in df.columns:
        st.error("Missing '__source_file' column - cannot split files properly.")
        return

    st.info("Download updated Excel files grouped by original source.")

    def to_excel_bytes(df_export):
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df_export.to_excel(writer, index=False)
        return output.getvalue()

    for source_file, group in df.groupby("__source_file"):
        export_df = group.drop(columns="__source_file", errors='ignore')
        excel_bytes = to_excel_bytes(export_df)
        st.download_button(
            label=f"Download {source_file} ({len(export_df)} controls)",
            data=excel_bytes,
            file_name=source_file,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    st.markdown("""
    ---
    ### Upload Instructions for Junior Staff
    1. Each downloaded file corresponds to one original control category.
    2. Upload each file back to the system it originated from – do not mix files.
    3. Contact your team lead if you encounter any issues.
    """)


def step_10_upload_guidance():
    st.header("Step 10: Upload Guidance for Junior Staff")

    if st.session_state['master_df'] is None:
        st.warning("Please complete prior steps first.")
        return

    df = st.session_state['master_df']

    if "__source_file" not in df.columns:
        st.error("Missing source file info ('__source_file'). Cannot display guidance.")
        return

    # Example: Define mapping of source file names to upload instructions
    upload_instructions = {
        "Controls_Category_A.xlsx": "Upload to Risk Management System under 'Category A'.",
        "Controls_Category_B.xlsx": "Upload to Investment Operations in 'Category B'.",
        "Controls_Category_C.xlsx": "Upload to Compliance system, 'Category C' folder."
    }

    source_files = df["__source_file"].unique()

    st.markdown("### Upload Instructions by File")

    for source_file in source_files:
        st.subheader(source_file)
        instruction = upload_instructions.get(
            source_file,
            "Please check with your supervisor for upload instructions related to this file."
        )
        st.write(instruction)

    st.markdown("""
    ---
    **Tips for Junior Staff:**
    - Download updated files from Step 9.
    - Always verify the correct destination for each file.
    - Reach out to your supervisor if unsure about file uploads.
    """)


# -------------------------- Navigation --------------------------

steps = {
    "1. Upload Data": step_1_upload_data,
    "2. Data Integration": step_2_data_integration,
    "3. Interactive Data Exploration": step_3_interactive_exploration,
    "4. Real-time Editing & Collaboration": step_4_real_time_collaboration,
    "5. Audit Trail Logging": step_5_audit_trail_logging,
    "6. Rules Management": step_6_rules_management,
    "7. Advanced Visualization": step_7_advanced_visualization,
    "8. Review & Resolve Flags": step_8_review_and_resolve_flags,
    "9. Export Data": step_9_export_data,
    "10. Upload Guidance": step_10_upload_guidance
}

st.sidebar.title("Navigation")
selected_step = st.sidebar.radio("Select Step", list(steps.keys()))

steps[selected_step]()
