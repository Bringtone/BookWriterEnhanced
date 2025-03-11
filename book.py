import os
import re
import io
import zipfile
import streamlit as st
from openai import OpenAI
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime

# ------------------------------
# 1. SETUP: Environment, API client, and Supabase
# ------------------------------
load_dotenv()  # Make sure your .env file contains SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY, STREAMLIT_APP_PASSWORD

API_KEY = os.environ.get("OPENAI_API_KEY")
VALID_PASSWORD = os.environ.get("STREAMLIT_APP_PASSWORD")
MODEL_NAME = "gpt-4o"  # Book generation uses gpt-4o

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY)

# Initialize Supabase client
from supabase import create_client, Client

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def save_book_metadata(category, title, english_pdf, tr_pdf, de_pdf, fr_pdf, es_pdf):
    """Inserts metadata for a generated book into the Supabase 'books' table."""
    data = {
        "category": category,
        "title": title,
        "english_pdf": english_pdf,
        "tr_pdf": tr_pdf,
        "de_pdf": de_pdf,
        "fr_pdf": fr_pdf,
        "es_pdf": es_pdf,
        "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    response = supabase.table("Books").insert(data).execute()
    return response


def get_created_books():
    """Fetches all books from the Supabase 'books' table as a DataFrame."""
    response = supabase.table("Books").select("*").execute()
    if response.data:
        return pd.DataFrame(response.data)
    else:
        return pd.DataFrame()


def count_words(text):
    """Returns the word count in the given text."""
    return len(re.findall(r'\w+', text))


def password_protect():
    """Simple password gate for Streamlit."""
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        st.title("Login")
        pwd = st.text_input("Enter the password:", type="password")
        if st.button("Log in"):
            if pwd == VALID_PASSWORD:
                st.session_state["logged_in"] = True
                st.rerun()
            else:
                st.error("Invalid password")
        st.stop()


password_protect()


def call_openai_chat_api(messages, model=MODEL_NAME, temperature=0.7):
    """Calls the OpenAI API using the provided client."""
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    return completion.choices[0].message.content.strip()


# ------------------------------
# 2. CORE BOOK-GENERATION LOGIC (unchanged)
# ------------------------------
def choose_chapter_count(desired_pages):
    estimated = desired_pages // 5
    if estimated < 5:
        return 5
    elif estimated > 20:
        return 20
    else:
        return estimated


def generate_outline(book_premise, desired_pages, chapter_count):
    system_message = {
        "role": "system",
        "content": (
            "You are an experienced author who creates professional, refined book outlines. "
            "Do not use special symbols (*, #, etc.) or bullet points. Write with a natural tone, "
            "like a real book. Number each chapter in a simple, clear manner."
        )
    }
    user_prompt = (
        f"User premise:\n{book_premise}\n\n"
        f"Please write a concise outline for a book with exactly {chapter_count} chapters, "
        f"aiming for ~{desired_pages} pages total. Number each chapter plainly (e.g., 'Chapter 1: Title'). "
        "Avoid repeating headings or using special characters. Keep it short and professional."
    )
    user_message = {"role": "user", "content": user_prompt}
    outline = call_openai_chat_api([system_message, user_message], model="gpt-4o")
    return outline


def generate_chapter(chapter_title, summary_so_far, book_premise, target_words=300):
    system_message = {
        "role": "system",
        "content": (
            "You are an experienced author writing a chapter of a book. "
            "Write in a professional, continuous style with multiple paragraphs. "
            "Avoid special symbols like *, #, or bullet points and do not repeat the chapter heading."
        )
    }
    user_prompt = (
        f"Book premise:\n{book_premise}\n\n"
        f"Summary of previous chapters:\n{summary_so_far}\n\n"
        f"Chapter title: '{chapter_title}'. "
        f"Please write around {target_words} words for this chapter in refined, engaging prose. "
        "Ensure the text flows naturally and does not simply restate the chapter title."
    )
    user_message = {"role": "user", "content": user_prompt}
    chapter_text = call_openai_chat_api([system_message, user_message], model="gpt-4o")

    if chapter_text.lower().startswith(chapter_title.lower()):
        chapter_text = chapter_text[len(chapter_title):].strip(":., \n")

    current_word_count = count_words(chapter_text)
    while current_word_count < target_words:
        remaining = target_words - current_word_count
        continue_prompt = (
            f"The chapter titled '{chapter_title}' is currently {current_word_count} words long. "
            f"Please continue writing to add approximately {remaining} more words. "
            "Keep the style consistent and avoid repeating previous content."
        )
        additional_text = call_openai_chat_api([system_message, {"role": "user", "content": continue_prompt}],
                                               model="gpt-4o")
        chapter_text += "\n" + additional_text.strip()
        current_word_count = count_words(chapter_text)

    return chapter_text.strip()


def save_as_pdf(chapters, filename="book_output.pdf"):
    c = canvas.Canvas(filename, pagesize=LETTER)
    width, height = LETTER
    margin_left = inch
    margin_top = height - inch

    for i, (title, text) in enumerate(chapters, start=1):
        c.setFont("Times-Roman", 18)
        c.drawString(margin_left, margin_top, title)
        y_pos = margin_top - 36
        c.setFont("Times-Roman", 12)
        max_chars_per_line = 90

        paragraphs = text.split("\n")
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            while len(paragraph) > max_chars_per_line:
                idx = paragraph.rfind(" ", 0, max_chars_per_line)
                if idx == -1:
                    idx = max_chars_per_line
                line = paragraph[:idx].strip()
                c.drawString(margin_left, y_pos, line)
                y_pos -= 14
                paragraph = paragraph[idx:].strip()
                if y_pos < inch:
                    c.showPage()
                    c.setFont("Times-Roman", 12)
                    y_pos = margin_top
            if paragraph:
                c.drawString(margin_left, y_pos, paragraph)
                y_pos -= 14
            y_pos -= 5
            if y_pos < inch:
                c.showPage()
                c.setFont("Times-Roman", 12)
                y_pos = margin_top

        c.showPage()

    c.save()
    return filename


# ------------------------------
# 3. TRANSLATION & MULTILINGUAL PDF GENERATION (unchanged except model for translation)
# ------------------------------
def translate_text(text, target_language):
    system_message = {
        "role": "system",
        "content": (
                "You are a professional translator. Translate the given text from English to " +
                target_language + ". Provide only the translated text."
        )
    }
    user_message = {"role": "user", "content": text}
    translated_text = call_openai_chat_api([system_message, user_message], model="gpt-4o-mini")
    return translated_text


def generate_translated_pdf(chapters, target_language, output_filename):
    translated_chapters = []
    for title, text in chapters:
        translated_title = translate_text(title, target_language)
        translated_text = translate_text(text, target_language)
        translated_chapters.append((translated_title, translated_text))
    pdf_file = save_as_pdf(translated_chapters, filename=output_filename)
    return pdf_file


# ------------------------------
# 4. BOOK GENERATION FROM GENERATED OUTLINE (no user confirmation)
# ------------------------------
def generate_book_from_outline(category, book_title, book_premise, desired_pages, outline):
    total_words = desired_pages * 300
    lines = outline.split("\n")
    chapter_lines = [line.strip() for line in lines if re.match(r"(?i)^chapter\s+\d+:", line.strip())]
    if not chapter_lines:
        chapter_count = choose_chapter_count(desired_pages)
        chapter_lines = [f"Chapter {i}: Untitled" for i in range(1, chapter_count + 1)]
    else:
        chapter_count = len(chapter_lines)
    words_per_chapter = max(200, total_words // chapter_count)

    chapters_data = []
    summary_so_far = ""
    for i, ch_title in enumerate(chapter_lines, start=1):
        st.write(f"Generating Chapter {i} for '{book_title}': {ch_title}")
        chapter_text = generate_chapter(ch_title, summary_so_far, book_premise, words_per_chapter)
        chapters_data.append((ch_title, chapter_text))
        summary_so_far += f"[{ch_title}] {chapter_text[:500]}...\n"

    english_pdf = f"{book_title}_EN.pdf"
    save_as_pdf(chapters_data, english_pdf)

    translations = {}
    for lang in ["TR", "DE", "FR", "ES"]:
        output_filename = f"{book_title}_{lang}.pdf"
        pdf_path = generate_translated_pdf(chapters_data, lang, output_filename)
        translations[lang] = pdf_path

    return english_pdf, translations


# ------------------------------
# 5. STREAMLIT APP PAGES
# ------------------------------
def show_created_books():
    st.header("Created Books")
    df = get_created_books()
    if df.empty:
        st.write("No books created yet.")
    else:
        st.dataframe(df)


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Generate Books", "Created Books"])

    if page == "Generate Books":
        st.title("Book Generator")
        st.subheader("Upload CSV File")
        uploaded_file = st.file_uploader("Upload CSV with columns: Category, Book Title, Content", type="csv")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, sep=None, engine='python')
                df.columns = df.columns.str.strip()
                required_columns = {"Category", "Book Title", "Content"}
                if required_columns.issubset(set(df.columns)):
                    st.write("CSV Preview:")
                    st.dataframe(df)

                    desired_pages = st.number_input("Approximate Page Count for each book:", min_value=1, max_value=999,
                                                    value=25)

                    if st.button("Generate All Books"):
                        generation_results = []
                        for idx, row in df.iterrows():
                            category = row["Category"]
                            book_title = row["Book Title"]
                            content = row["Content"]
                            chapter_count = choose_chapter_count(desired_pages)
                            outline = generate_outline(content, desired_pages, chapter_count)
                            st.write(f"Outline generated for '{book_title}'.")
                            english_pdf, translations = generate_book_from_outline(category, book_title, content,
                                                                                   desired_pages, outline)
                            save_book_metadata(category, book_title, english_pdf,
                                               translations["TR"], translations["DE"], translations["FR"],
                                               translations["ES"])
                            generation_results.append({
                                "category": category,
                                "book_title": book_title,
                                "english_pdf": english_pdf,
                                "TR_pdf": translations["TR"],
                                "DE_pdf": translations["DE"],
                                "FR_pdf": translations["FR"],
                                "ES_pdf": translations["ES"]
                            })
                        st.session_state["generation_results"] = generation_results
                        st.success("All books have been generated!")

                        st.header("Download a Single ZIP of All Generated Books")
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                            for res in generation_results:
                                pdf_files = {
                                    "EN": res["english_pdf"],
                                    "TR": res["TR_pdf"],
                                    "DE": res["DE_pdf"],
                                    "FR": res["FR_pdf"],
                                    "ES": res["ES_pdf"]
                                }
                                for lang, path in pdf_files.items():
                                    arcname = f"{res['book_title']}_{lang}.pdf"
                                    zipf.write(path, arcname=arcname)
                        zip_buffer.seek(0)
                        st.download_button(
                            "Download ALL Books (Single ZIP)",
                            zip_buffer,
                            file_name="All_Books.zip",
                            mime="application/zip"
                        )
                else:
                    st.error("CSV must contain columns: Category, Book Title, Content")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
    elif page == "Created Books":
        show_created_books()


if __name__ == "__main__":
    main()
