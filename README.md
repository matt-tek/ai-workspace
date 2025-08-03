# AI-gen
test on ai technologies to level up 

```mermaid
graph TD;
  subgraph Client
    A1[Utilisateur (Web/App)]
    A2[App frontend (Vue.js / React / Streamlit)]
  end

  subgraph FastAPI Backend
    B1[API Router /main.py]
    B2[meals.py]
    B3[shopping.py]
    B4[user.py]

    B5[Agent: smartmeal.py]
    B6[Planner: planner.py]
    B7[Interpreter: interpreter.py]
    B8[RecipeProvider: recipe_provider.py]
    B9[Memory Manager: memory.py]

    B10[Services: OpenAI / Spoonacular]
    B11[Database: Supabase/PostgreSQL]

    A2 -->|HTTP Request| B1
    B1 --> B2
    B1 --> B3
    B1 --> B4

    B2 --> B5
    B5 --> B7
    B5 --> B6
    B6 --> B8
    B5 --> B9
    B5 --> B10

    B4 --> B11
    B9 --> B11
  end

  subgraph External APIs
    C1[OpenAI / LLMs]
    C2[Spoonacular / Recipes API]
  end

  B10 --> C1
  B10 --> C2

```