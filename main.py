import numpy as np
from PIL import Image
import streamlit as st
import fitz, faiss, os, io, openai
from transformers import CLIPProcessor, CLIPModel
import requests


st.set_page_config(page_icon="😶‍🌫️", page_title="QA", layout="wide")
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

header = st.container()
header.title("Query Assistance Genie")
header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)
st.markdown(
    """
    <style>
        .st-emotion-cache-vj1c9o {
            background-color: rgb(242, 242, 242, 0.68);
        }
        div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
            position: sticky;
            background-color: rgb(242, 242, 242, 0.68);
            z-index: 999;
            text-align: center;
        }
        .fixed-header {
            border-bottom: 0;
        }
      div[data-testid="stVerticalBlock"] div:has(div.fixed-header) .st-emotion-cache-1wmy9hl {
            display: flex;
            flex-direction: column;
            margin-top: -70px;
        }
       
        h1 {
        font-family: "Source Sans Pro", sans-serif;
        font-weight: 500;
        color: rgb(49, 51, 63);
        padding: 1.25rem 0px 1rem;
        margin: 0px;
        line-height: 1.2;
        color: black;
        border:1px solid black;
    }
   
    .st-emotion-cache-jkfxgf p {
        word-break: break-word;
        margin-bottom: 0px;
        font-size: 16px;
        font-weight: 600;
        color : purple;
    }
    .st-emotion-cache-1puwf6r p {
    word-break: break-word;
    margin-bottom: 0px;
    font-size: 14px;
    font-weight: 600;
    }
    .st-b6 {
    border-bottom-color: black;
    }
    .st-b5 {
        border-top-color: black;
    }
    .st-b4 {
        border-right-color: black;
    }
    .st-b3 {
        border-left-color: black;
    }
    .st-emotion-cache-1igbibe{
        border: 1px solid black;
    }
        .st-emotion-cache-kn8v7q {
        display: flex;
        flex-flow: column;
        max-width: 40%;
        align-items: stretch;
        flex-direction: row;
        justify-content: center;
    }
    </style>
       """,
       unsafe_allow_html=True
)
 
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)




api_key = st.secrets["OPENAI_API_KEY"]

# Environment setup
openai.api_key = api_key

# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def extract_content_and_images_with_context(pdf_path):
    doc = fitz.open(pdf_path)
    data = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        images = []

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            try:
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                images.append(image)
            except Exception as e:
                print(f"Error processing image on page {page_num}, image {img_index}: {e}")
        
        # Extract surrounding text as context
        surrounding_paragraphs = "\n".join(text.split("\n")[:10])  # Take first 5 lines or adjust as needed
        data.append({
            "page_num": page_num,
            "text": text,
            "images": [{"image": image, "context": surrounding_paragraphs} for image in images]
        })
    return data

def get_clip_embedding(text=None, image=None):
    """
    Generate CLIP embedding for text or image.
    Returns a numpy array of embeddings.
    """
    if text is not None:
        text_input = clip_processor(text=[text[:77]], return_tensors="pt")
        embedding = clip_model.get_text_features(**text_input)
    elif image is not None:
        image_input = clip_processor(images=image, return_tensors="pt")
        embedding = clip_model.get_image_features(**image_input)
    else:
        raise ValueError("Either text or image must be provided")
    
    return embedding.detach().numpy().flatten()

def setup_faiss_indices(data):
    text_embeddings = []
    image_embeddings = []
    metadata = []

    for item in data:
        # Process text
        if item.get("text", "").strip():
            try:
                text_embedding = get_clip_embedding(text=item["text"])
                text_embeddings.append(text_embedding)
            except Exception as e:
                print(f"Error processing text embedding: {e}")

        # Process images with context
        for image_data in item.get("images", []):
            try:
                image_embedding = get_clip_embedding(image=image_data["image"])
                combined_context = f"Page {item['page_num']} Context: {image_data['context']}"
                text_embedding = get_clip_embedding(text=combined_context)
                
                # Combine embeddings
                combined_embedding = (image_embedding + text_embedding) / 2
                
                metadata.append({
                    "page_num": item["page_num"],
                    "image": image_data["image"],
                    "text_context": combined_context
                })
                
                image_embeddings.append(combined_embedding)
            except Exception as e:
                print(f"Error processing image embedding: {e}")

    # Create Faiss indices
    text_index, image_index = None, None

    if text_embeddings:
        text_embeddings = np.array(text_embeddings)
        text_dim = text_embeddings.shape[-1]
        text_index = faiss.IndexFlatL2(text_dim)
        text_index.add(text_embeddings)

    if image_embeddings:
        image_embeddings = np.array(image_embeddings)
        image_dim = image_embeddings.shape[-1]
        image_index = faiss.IndexFlatL2(image_dim)
        image_index.add(image_embeddings)

    return text_index, image_index, metadata

def query_faiss(query, text_index, image_index, metadata, k=2):
    # Generate query embedding
    query_embedding = get_clip_embedding(text=query)

    # Ensure query embedding is 2D for Faiss search
    query_embedding = query_embedding.reshape(1, -1)

    # Search indices
    text_D, text_I = text_index.search(query_embedding, k)
    image_D, image_I = image_index.search(query_embedding, k)

    results = {
        "text": [{"score": text_D[0][i], "index": text_I[0][i]} for i in range(k)],
        "images": [{"score": image_D[0][i], "metadata": metadata[image_I[0][i]]} for i in range(k)]
    }
    return results

def generate_response(query, text_results, image_results):
    text_context = "\n".join([f"Text Result {i+1}: {r['text']}" for i, r in enumerate(text_results)])
    image_descriptions = "\n".join([f"Image {i+1}: From page {r['metadata']['page_num']} with context: {r['metadata']['text_context']}" for i, r in enumerate(image_results)])

    prompt = f"""
    Query: {query}

    Text Context:
    {text_context}

    Available Images:
    {image_descriptions}

    Answer based on the context:
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an assistant specializing in analyzing PDFs containing text and images."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"Error generating response: {str(e)}"
        
def set_query(query):
        st.session_state.query_input = query
        st.rerun()

def main():
    
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""

    # The text input field, populated with the query stored in session state
    query = st.text_input("Enter your query:", value=st.session_state.query_input)

    st.write("Suggested Queries:")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Compare the architectural differences between the AMD Ryzen 9 9950X and its predecessor, the Ryzen 9 7950X, focusing on core architecture, thread count, chip design, and performance optimizations. Show detailed diagrams highlighting these improvements."):
            set_query("Compare the architectural differences between the AMD Ryzen 9 9950X and its predecessor, the Ryzen 9 7950X, focusing on core architecture, thread count, chip design, and performance optimizations. Show detailed diagrams highlighting these improvements.")
    with col2:
            
        if st.button("Visualize the power consumption and thermal performance differences between the Ryzen 9 9950X and the Ryzen 9 7950X under an all-core workload. Include charts displaying temperature, power draw, and thermal throttling."):
            set_query("Visualize the power consumption and thermal performance differences between the Ryzen 9 9950X and the Ryzen 9 7950X under an all-core workload. Include charts displaying temperature, power draw, and thermal throttling.")

    with col3:
                
        if st.button("Highlight the significance of the AMD Ryzen 9 9950X’s AVX-512 and VNNI support for machine learning and AI workloads. Illustrate how these features impact performance in deep learning tasks."):
            set_query("Highlight the significance of the AMD Ryzen 9 9950X’s AVX-512 and VNNI support for machine learning and AI workloads. Illustrate how these features impact performance in deep learning tasks.")

    with col4:

        if st.button("Identify the reasons why the AMD Ryzen 9 9950X might be preferred for productivity tasks over gaming, showcasing its strengths in multi-threaded workloads and creative applications like video rendering, while balancing gaming performance."):
            set_query("Identify the reasons why the AMD Ryzen 9 9950X might be preferred for productivity tasks over gaming, showcasing its strengths in multi-threaded workloads and creative applications like video rendering, while balancing gaming performance.")
    
    if query:
        with st.spinner("Generating Response"):
            # PDF path
            pdf_path = r"link.pdf"

            # Extract content and setup indices
            data = extract_content_and_images_with_context(pdf_path)
            text_index, image_index, metadata = setup_faiss_indices(data)
            try:
                results = query_faiss(query, text_index, image_index, metadata)
                text_results = [data[r["index"]] for r in results["text"]]
                image_results = results["images"]
                with st.expander("Response"):
                    response = generate_response(query, text_results, image_results)
                    st.write(response)

                with st.expander("Related Images"):
                    for r in image_results:
                        st.image(r["metadata"]["image"], caption=f"Page {r['metadata']['page_num']}: {r['metadata']['text_context']}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
