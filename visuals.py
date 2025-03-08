import streamlit as st
from neo4j import GraphDatabase
from pyvis.network import Network
import os
import tempfile

class Neo4jConnection:
    def __init__(self, uri, username, password):
        self.driver = None
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            # Test the connection
            self.driver.verify_connectivity()
            st.success("Successfully connected to Neo4j database!")
        except Exception as e:
            st.error(f"Failed to connect to Neo4j database: {str(e)}")
            st.info("""
            Please check:
            1. Neo4j server is running
            2. Credentials are correct
            3. Database URL is accessible
            """)
            raise e

    def close(self):
        if self.driver:
            self.driver.close()

    def query(self, query):
        data = []
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                data.append(record)
        return data

def create_graph_visualization(data):
    """Create an interactive graph visualization using PyVis"""
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
    
    # Add physics options for better visualization
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100)
    net.show_buttons(filter_=['physics'])
    
    # Track unique nodes to avoid duplicates
    added_nodes = set()
    
    for record in data:
        source = str(record["source"]) if record["source"] is not None else "Unknown"
        target = str(record["target"]) if record["target"] is not None else "Unknown"
        relation = str(record["relation"])
        
        # Add nodes if they don't exist
        if source not in added_nodes:
            net.add_node(source, label=source, title=source, color="#00ff1e")
            added_nodes.add(source)
        if target not in added_nodes:
            net.add_node(target, label=target, title=target, color="#162347")
            added_nodes.add(target)
            
        # Add edge with relationship type
        net.add_edge(source, target, title=relation, arrows="to")
    
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
        net.save_graph(tmp_file.name)
        return tmp_file.name

def main():
    st.set_page_config(page_title="Neo4j Graph Visualization", layout="wide")
    
    st.title("Neo4j Graph Visualization")
    
    # Database connection settings
    with st.sidebar:
        st.header("Database Settings")
        neo4j_uri = st.text_input("Neo4j URI", "neo4j+s://70aaffdf.databases.neo4j.io:7687")
        neo4j_user = st.text_input("Username", "neo4j")
        neo4j_password = st.text_input("Password", "h399A3mepG_hj_tiQOPwaq9ufzAEGXeFvnqWhFrXPvQ")

        
        st.header("Sample Queries")
        st.code("""
        # Get all relationships
        MATCH (n)-[r]->(m) 
        RETURN n.name AS source, m.name AS target, type(r) AS relation
        LIMIT 25
        
        # Get specific node relationships
        MATCH (n)-[r]->(m) 
        WHERE n.name = 'YourNodeName'
        RETURN n.name AS source, m.name AS target, type(r) AS relation
        """)
    
    # Query input
    cypher_query = st.text_area(
        "Enter Cypher Query",
        value="MATCH (n)-[r]->(m) RETURN COALESCE(n.name, n.id, ID(n)) AS source, COALESCE(m.name, m.id, ID(m)) AS target,  type(r) AS relation LIMIT 173"
    )
    
    # Connect and visualize button
    if st.button("Run Query and Visualize"):
        try:
            # Initialize connection
            conn = Neo4jConnection(neo4j_uri, neo4j_user, neo4j_password)
            
            with st.spinner("Fetching data from Neo4j..."):
                # Execute query
                graph_data = conn.query(cypher_query)
                # st.write("Sample of returned data:")
                # for i, record in enumerate(graph_data[:5]):  # Show first 5 records
                #     st.write(f"Record {i+1}:", dict(record))
                                
                if not graph_data:
                    st.warning("No data returned from the query.")
                    return
                
                st.success(f"Found {len(graph_data)} relationships!")
                
            with st.spinner("Generating visualization..."):
                # Create and display graph
                graph_file = create_graph_visualization(graph_data)
                with open(graph_file, 'r', encoding='utf-8') as f:
                    html_data = f.read()
                st.components.v1.html(html_data, height=800)
                
                # Cleanup temporary file
                os.unlink(graph_file)
            
            # Close connection
            conn.close()
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")



if __name__ == "__main__":
    main()

# # First, let's modify your code to include some error handling and debugging
# from neo4j import GraphDatabase
# import socket

# # Your AuraDB credentials
# URI = "neo4j+s://70aaffdf.databases.neo4j.io:7687"
# USER = "neo4j"
# PASSWORD = "h399A3mepG_hj_tiQOPwaq9ufzAEGXeFvnqWhFrXPvQ"


# # Then try the connection
# try:
#     driver = GraphDatabase.driver(
#         URI, 
#         auth=(USER, PASSWORD),
#         max_connection_lifetime=60  # Reduce connection lifetime
#     )
#     driver.verify_connectivity()
#     print("Connection successful!")
    
#     with driver.session() as session:
#         result = session.run("RETURN 1 AS num")
#         print(result.single()[0])
        
# except Exception as e:
#     print(f"Connection failed: {str(e)}")
# finally:
#     if 'driver' in locals():
#         driver.close()