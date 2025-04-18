import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="🔍 Analyse Utilisateur",
    page_icon="🎯",
    layout="wide"
)

# Chargement des données
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('final_df.csv')
        # Convertir les colonnes problématiques en string
        for col in ['user_name', 'user_email', 'country', 'host', 'medium']:
            if col in df.columns:
                df[col] = df[col].astype(str)
        return df
    except FileNotFoundError:
        st.error("Fichier de données non trouvé. Veuillez vérifier le chemin du fichier.")
        st.stop()

# Chargement du modèle de prédiction
@st.cache_resource
def load_model():
    try:
        model = joblib.load('rf_pipeline.pkl')
        return model
    except FileNotFoundError:
        st.warning("Modèle de prédiction non trouvé. Certaines fonctionnalités de prédiction seront désactivées.")
        return None

# Chargement des données et du modèle
df = load_data()
model = load_model()

# Chargement des modèles de clustering
main_pipeline = joblib.load("main_clustering_pipeline.pkl")
sub_kmeans = joblib.load("sub_clustering_model.pkl")
num_cols = [
    'nb_sessions','avg_pageviews','nb_clicks','clicks_per_session','click_frequency',
    'recency_days','avg_days_between_sessions','avg_prior_sessions','session_efficiency',
    'engagement_score','global_engagement_score','engagement_intensity',
    'multi_interaction_score','engagement_trend','retention_like_score','executive_weighted_score',
    'bounce_rate','nb_entry_pages','entry_pages_per_session','unique_clicks_ratio',
    'pageview_click_ratio','diversity_score','regularity_score'
]

# Appliquer le pipeline principal
df['cluster'] = main_pipeline.predict(df[num_cols]).astype(str)

# Identifier les points du cluster 3
mask_cluster_3 = df['cluster'] == '3'
if mask_cluster_3.sum() > 0:
    # Prétraiter les données du cluster 3
    X_cluster_3 = df.loc[mask_cluster_3, num_cols]
    X_cluster_3_processed = main_pipeline.named_steps['prep'].transform(X_cluster_3)
    
    # Appliquer le sous-clustering
    sub_labels = sub_kmeans.predict(X_cluster_3_processed)
    
    # Récupérer les indices des points du cluster 3
    cluster_3_indices = df[mask_cluster_3].index.to_numpy()
    
    # Pour chaque sous-cluster, mettre à jour les étiquettes
    for s in range(2):  # 2 sous-clusters
        sub_idx = cluster_3_indices[sub_labels == s]
        df.loc[sub_idx, 'cluster'] = f"3_{s+1}"

# Dictionnaire des profils de clusters détaillés pour Management & Data Science
cluster_profiles = {
    '0': {
        'title': "Faiblement Engagé, Passif",
        'description': "Utilisateur avec un faible engagement et peu d'interactions, principalement issu de recherche organique.",
        'interests': ["Concepts fondamentaux de data science", "Introduction à l'analyse de données", "Tutoriels de base"],
        'recommendations': [
            "Série 'Data Science pour tous' - Introduction aux concepts fondamentaux",
            "Guide étape par étape pour débuter en analyse de données",
            "Webinaire d'introduction : Les bases de la data science en entreprise"
        ],
        'content_types': ["Articles introductifs", "Vidéos explicatives", "Infographies simplifiées"],
        'email_template': {
            'subject': "🔍 Découvrez les fondamentaux de la Data Science en 5 minutes",
            'content': "Bonjour,\n\nNous avons sélectionné pour vous une introduction aux concepts fondamentaux de la Data Science, adaptée à votre niveau.\n\nDécouvrez notre série de contenus 'Data Science pour tous' et commencez votre parcours d'apprentissage dès aujourd'hui."
        }
    },
    '1': {
        'title': "Passif à Risque élevé de Désengagement",
        'description': "Utilisateur avec un engagement très limité et des interactions modérées sans profondeur.",
        'interests': ["Cas d'études sectoriels", "Applications pratiques", "Contenus courts et accessibles"],
        'recommendations': [
            "Cas d'étude : Impact de la data science dans votre secteur",
            "Mini-formation : 15 minutes par jour pour progresser en data science",
            "Sélection personnalisée d'articles courts basés sur vos centres d'intérêt"
        ],
        'content_types': ["Cas d'études", "Tutoriels courts", "Newsletters ciblées"],
        'email_template': {
            'subject': "🔄 Voici ce que vous avez manqué sur Management & Data Science",
            'content': "Bonjour,\n\nDepuis votre dernière visite, notre communauté a partagé de nombreux contenus qui pourraient vous intéresser.\n\nVoici une sélection personnalisée basée sur vos centres d'intérêt :\n\n- Cas d'étude : Impact de la Data Science dans votre secteur\n- Nouveaux datasets disponibles pour vos analyses\n- Discussions actives sur les sujets qui vous passionnent"
        }
    },
    '2': {
        'title': "Très Faible Engagement",
        'description': "Utilisateur avec un engagement très faible et peu d'interactions.",
        'interests': ["Contenus simplifiés", "Success stories", "Applications concrètes"],
        'recommendations': [
            "Data Science simplifiée : comprendre les concepts clés en 5 minutes",
            "Success stories : comment la data science a transformé ces entreprises",
            "Accès premium temporaire : découvrez nos contenus exclusifs"
        ],
        'content_types': ["Vidéos courtes", "Infographies", "Témoignages"],
        'email_template': {
            'subject': "🎁 Accès premium offert : Redécouvrez Management & Data Science",
            'content': "Bonjour,\n\nNous vous offrons un accès temporaire à tous nos contenus premium pour vous permettre de redécouvrir la valeur de Management & Data Science.\n\nVoici ce que vous pourrez explorer :\n\n- Études de cas complètes avec code source\n- Datasets exclusifs pour vos analyses\n- Formations vidéo sur les techniques avancées"
        }
    },
    '3_1': {
        'title': "Actif avec Fort Potentiel",
        'description': "Utilisateur avec un bon engagement et une bonne interactivité, prêt à approfondir ses connaissances.",
        'interests': ["Data challenges", "Projets collaboratifs", "Contenus avancés"],
        'recommendations': [
            "Data Challenge : Analyse prédictive sur données réelles",
            "Webinaire avancé : Techniques de machine learning pour professionnels",
            "Parcours thématique : Spécialisation en NLP pour data scientists"
        ],
        'content_types': ["Projets pratiques", "Webinaires avancés", "Forums spécialisés"],
        'email_template': {
            'subject': "🏆 Rejoignez notre programme Data Leaders et partagez votre expertise",
            'content': "Bonjour,\n\nVotre activité sur Management & Data Science vous place parmi nos utilisateurs les plus engagés.\n\nNous vous invitons à rejoindre notre programme Data Leaders qui vous permettra de :\n\n- Participer à nos Data Challenges exclusifs\n- Obtenir des badges de reconnaissance pour vos contributions\n- Accéder en avant-première à nos nouveaux contenus"
        }
    },
    '3_2': {
        'title': "Moyennement Actif mais Sous-exploité",
        'description': "Utilisateur avec un engagement moyen et des interactions moyennes, avec un potentiel inexploité.",
        'interests': ["Datasets pratiques", "Discussions thématiques", "Mini-challenges"],
        'recommendations': [
            "Sélection de datasets pour pratiquer vos compétences",
            "Communauté : rejoignez les discussions sur l'IA générative",
            "Mini-challenge : optimisation d'algorithmes de clustering"
        ],
        'content_types': ["Datasets commentés", "Forums de discussion", "Exercices pratiques"],
        'email_template': {
            'subject': "📊 Ressources recommandées pour approfondir vos connaissances",
            'content': "Bonjour,\n\nBasé sur votre activité récente, nous avons sélectionné des ressources qui pourraient vous aider à approfondir vos connaissances en data science.\n\nVoici nos recommandations personnalisées :\n\n- Articles intermédiaires sur les techniques de clustering\n- Datasets récents dans votre domaine\n- Mini-challenge adapté à votre niveau"
        }
    },
    '4': {
        'title': "Bonne Interaction Générale",
        'description': "Utilisateur avec un engagement élevé mais des interactions moyennes, prêt à diversifier ses connaissances.",
        'interests': ["Nouveaux domaines", "Partage de connaissances", "Collaborations"],
        'recommendations': [
            "Découvrez de nouveaux domaines : Data Science pour la finance",
            "Atelier pratique : partager vos connaissances avec la communauté",
            "Collaborations : projets ouverts recherchant des contributeurs"
        ],
        'content_types': ["Articles spécialisés", "Ateliers pratiques", "Projets collaboratifs"],
        'email_template': {
            'subject': "🌟 Explorez de nouveaux domaines de la Data Science",
            'content': "Bonjour,\n\nVotre engagement sur Management & Data Science est excellent, et nous pensons que vous pourriez être intéressé par l'exploration de nouveaux domaines connexes.\n\nVoici des suggestions basées sur votre profil :\n\n- Data Science pour la Finance: Articles et ressources\n- NLP: Projets pratiques pour développer vos compétences\n- Opportunités de partager votre expertise dans vos domaines de prédilection"
        }
    },
    '5': {
        'title': "Petit Groupe Actif",
        'description': "Utilisateur engagé avec un bon potentiel d'approfondissement, intéressé par des contenus de niche.",
        'interests': ["Contenus spécialisés", "Groupes d'experts", "Publications"],
        'recommendations': [
            "Contenu spécialisé : Techniques avancées de feature engineering",
            "Groupe d'experts : rejoignez la discussion sur l'éthique en IA",
            "Publication : opportunités de partager votre expertise"
        ],
        'content_types': ["Articles de niche", "Groupes spécialisés", "Opportunités de publication"],
        'email_template': {
            'subject': "👑 Invitation à rejoindre notre groupe d'experts",
            'content': "Bonjour,\n\nVotre expertise et votre engagement sur Management & Data Science vous distinguent.\n\nNous vous invitons à rejoindre notre groupe d'experts où vous pourrez :\n\n- Participer à des discussions spécialisées avec d'autres experts\n- Contribuer à la création de contenus de niche\n- Être reconnu comme expert dans votre domaine"
        }
    },
    '6': {
        'title': "Très Actif et Intense",
        'description': "Utilisateur très actif avec une forte intensité d'engagement et des interactions très élevées.",
        'interests': ["Accès anticipé", "Mentorat", "Événements exclusifs"],
        'recommendations': [
            "Accès anticipé : nouvelles fonctionnalités de la plateforme",
            "Programme de mentorat : devenez animateur thématique",
            "Événement exclusif : rencontre avec des experts du domaine"
        ],
        'content_types': ["Contenus exclusifs", "Événements VIP", "Opportunités de mentorat"],
        'email_template': {
            'subject': "🔑 Accès VIP : Programme Data Science Influencers",
            'content': "Bonjour,\n\nFélicitations ! Votre activité exceptionnelle sur Management & Data Science vous qualifie pour notre programme VIP 'Data Science Influencers'.\n\nVos avantages exclusifs :\n\n- Accès anticipé aux nouvelles fonctionnalités et contenus\n- Invitations à des événements privés avec des experts du domaine\n- Opportunités de collaboration sur des projets stratégiques\n- Badge VIP visible sur votre profil"
        }
    },
    '7': {
        'title': "Petit Groupe Ultra-Engagé",
        'description': "Utilisateur d'élite avec un engagement exceptionnel et une très forte intensité, ambassadeur potentiel.",
        'interests': ["Co-création", "Ambassadeur", "Événements VIP"],
        'recommendations': [
            "Programme Ambassadeur : co-création de contenus premium",
            "Interview : partagez votre expertise avec notre communauté",
            "Événement VIP : table ronde sur l'avenir de la data science"
        ],
        'content_types': ["Co-création de contenu", "Interviews", "Tables rondes"],
        'email_template': {
            'subject': "🌠 Devenez Ambassadeur Management & Data Science",
            'content': "Bonjour,\n\nVous faites partie du cercle très restreint de nos utilisateurs les plus engagés et nous souhaitons vous proposer une relation privilégiée avec notre équipe.\n\nEn tant qu'Ambassadeur, vous bénéficierez de :\n\n- Un canal de communication direct avec notre équipe de direction\n- Des opportunités de co-création de contenus et fonctionnalités\n- Une mise en avant de votre expertise via des interviews et témoignages\n- Des avantages exclusifs réservés aux Ambassadeurs"
        }
    }
}

# En-tête de la page
st.title("🔍 Analyse Utilisateur Individuelle")
st.markdown("### Exploration détaillée du profil d'un utilisateur de Management & Data Science")

# Bannière de fonctionnalités à venir
st.info("🚀 **Nouvelles fonctionnalités à venir :** Analyse prédictive avancée, Parcours utilisateur détaillé et Recommandations IA personnalisées. Restez connectés !")

# Sélection de l'utilisateur
col1, col2 = st.columns([1, 3])

with col1:
    # Option de recherche par ID ou par filtres
    search_method = st.radio(
        "Méthode de recherche",
        ["ID Utilisateur", "Filtres avancés"]
    )
    
    if search_method == "ID Utilisateur":
        # Conversion explicite en string pour éviter les problèmes
        visitor_ids = df['visitor_id'].astype(str).tolist()
        user_id = st.selectbox(
            "Sélectionnez un ID utilisateur",
            visitor_ids
        )
        # Utilisation de la comparaison de strings
        user_data = df[df['visitor_id'].astype(str) == user_id].iloc[0]
    
    else:  # Filtres avancés
        # Filtres pour trouver un utilisateur
        cluster_filter = st.selectbox(
            "Cluster",
            sorted(df['cluster'].unique()),
            format_func=lambda x: f"Cluster {x} - {cluster_profiles.get(x, {}).get('title', 'Non défini')}"
        )
        
        engagement_filter = st.slider(
            "Score d'engagement minimum",
            min_value=0,
            max_value=100,
            value=50
        )
        
        is_active_filter = st.checkbox("Utilisateurs actifs uniquement", value=False)
        
        # Application des filtres
        filtered_df = df[df['cluster'] == cluster_filter]
        filtered_df = filtered_df[filtered_df['engagement_score'] >= engagement_filter]
        
        if is_active_filter:
            filtered_df = filtered_df[filtered_df['is_active'] == 1]
        
        if len(filtered_df) > 0:
            # Conversion en string pour éviter les problèmes
            visitor_ids = filtered_df['visitor_id'].astype(str).tolist()
            user_id = st.selectbox(
                "Sélectionnez un utilisateur parmi les résultats filtrés",
                visitor_ids
            )
            user_data = df[df['visitor_id'].astype(str) == user_id].iloc[0]
        else:
            st.warning("Aucun utilisateur ne correspond aux critères sélectionnés.")
            st.stop()

# Affichage des informations utilisateur
st.subheader(f"👤 Profil de l'utilisateur: {user_data['visitor_id']}")

# Métriques principales avec couleurs adaptées
col1, col2, col3, col4 = st.columns(4)

with col1:
    cluster_label = user_data['cluster']
    cluster_title = cluster_profiles.get(cluster_label, {}).get('title', 'Non défini')
    st.metric("Segment", f"{cluster_label} - {cluster_title}")

with col2:
    engagement_score = user_data['engagement_score']
    engagement_delta = engagement_score - df['engagement_score'].mean()
    st.metric("Score d'engagement", f"{engagement_score:.2f}", f"{engagement_delta:.2f} vs moyenne")

with col3:
    status = "Actif" if user_data['is_active'] == 1 else "Inactif"
    status_icon = "✅" if user_data['is_active'] == 1 else "❌"
    st.metric("Statut", f"{status_icon} {status}")

with col4:
    risk = "Élevé" if user_data['will_disengage_30d'] == 1 else "Faible"
    risk_icon = "⚠️" if user_data['will_disengage_30d'] == 1 else "✅"
    st.metric("Risque de désengagement", f"{risk_icon} {risk}")

# Résumé du profil utilisateur
user_cluster = user_data['cluster']
if user_cluster in cluster_profiles:
    profile = cluster_profiles[user_cluster]
    st.info(f"**Profil utilisateur :** {profile['description']}")

# Onglets pour organiser les informations
tabs = st.tabs(["📊 Comportement", "👥 Profil détaillé", "🔮 Prédictions", "📈 Historique", "🎯 Actions recommandées", "🚀 Fonctionnalités à venir"])

with tabs[0]:  # Comportement
    col1, col2 = st.columns(2)
    
    with col1:
        # Tableau des métriques d'engagement
        st.subheader("Métriques d'engagement")
        engagement_metrics = [
            'engagement_score', 'global_engagement_score', 'engagement_intensity',
            'multi_interaction_score', 'engagement_trend', 'session_efficiency',
            'retention_like_score', 'executive_weighted_score'
        ]
        
        engagement_df = pd.DataFrame({
            'Métrique': engagement_metrics,
            'Valeur': [user_data[metric] for metric in engagement_metrics]
        })
        
        st.dataframe(engagement_df, use_container_width=True)
        
        # Informations sur les sessions
        st.subheader("🖥️ Activité sur la plateforme")
        
        session_metrics = [
            'nb_sessions', 'avg_pageviews', 'bounce_rate', 'recency_days',
            'avg_days_between_sessions', 'nb_clicks', 'nb_unique_elements_clicked',
            'nb_entry_pages'
        ]
        
        session_df = pd.DataFrame({
            'Métrique': session_metrics,
            'Valeur': [user_data[metric] for metric in session_metrics]
        })
        
        st.dataframe(session_df, use_container_width=True)
    
    with col2:
        # Comparaison avec la moyenne du cluster
        st.subheader(f"Comparaison avec le segment {user_cluster}")
        
        # Prendre uniquement les colonnes numériques pour éviter l'erreur
        num_cols = df.select_dtypes(include=[np.number]).columns
        cluster_avg = df[df['cluster'] == user_cluster][num_cols].mean()
        
        comparison_metrics = [
            'engagement_score', 'global_engagement_score', 'engagement_intensity',
            'multi_interaction_score', 'session_efficiency'
        ]
        
        comparison_df = pd.DataFrame({
            'Métrique': comparison_metrics,
            'Utilisateur': [user_data[metric] for metric in comparison_metrics],
            'Moyenne Segment': [cluster_avg.get(metric, 0) for metric in comparison_metrics]
        })
        
        # Création d'un graphique à barres pour la comparaison
        fig = px.bar(
            comparison_df,
            x='Métrique',
            y=['Utilisateur', 'Moyenne Segment'],
            barmode='group',
            title=f"Comparaison avec la moyenne du segment {user_cluster}",
            color_discrete_sequence=['#3366CC', '#DC3912'],
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Graphique radar pour visualiser le profil utilisateur
        st.subheader("Profil d'engagement")
        categories = comparison_metrics
        
        # Normalisation des valeurs pour le radar chart
        user_values = [float(user_data[cat]) for cat in categories]
        cluster_values = [float(cluster_avg.get(cat, 0)) for cat in categories]
        
        # Trouver le maximum pour chaque métrique dans tout le dataset
        max_values = [float(df[cat].max()) if cat in df.columns else 1 for cat in categories]
        
        # Normaliser les valeurs
        user_normalized = [user_values[i]/max_values[i] if max_values[i] > 0 else 0 for i in range(len(categories))]
        cluster_normalized = [cluster_values[i]/max_values[i] if max_values[i] > 0 else 0 for i in range(len(categories))]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=user_normalized,
            theta=categories,
            fill='toself',
            name='Utilisateur',
            line=dict(color='#3366CC')
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=cluster_normalized,
            theta=categories,
            fill='toself',
            name=f'Moyenne Segment {user_cluster}',
            line=dict(color='#DC3912')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tabs[1]:  # Profil détaillé
    col1, col2 = st.columns(2)
    
    with col1:
        # Informations utilisateur - Gestion sécurisée des valeurs
        st.subheader("Informations personnelles")
        user_info = {}
        
        # Traitement sécurisé de user_name
        if pd.notna(user_data['user_name']):
            user_name = str(user_data['user_name'])
            user_info['Nom'] = user_name if user_name != '(Visiteur)' else 'Non renseigné'
        else:
            user_info['Nom'] = 'Non renseigné'
        
        # Traitement sécurisé de user_email
        if pd.notna(user_data['user_email']):
            user_email = str(user_data['user_email'])
            user_info['Email'] = user_email if user_email != '(not set)' else 'Non renseigné'
        else:
            user_info['Email'] = 'Non renseigné'
        
        # Traitement sécurisé de country
        if pd.notna(user_data['country']):
            country = str(user_data['country'])
            user_info['Pays'] = country if country != '(not set)' else 'Non renseigné'
        else:
            user_info['Pays'] = 'Non renseigné'
        
        # Traitement sécurisé de host
        if pd.notna(user_data['host']):
            host = str(user_data['host'])
            user_info['Fournisseur'] = host if host != '(not set)' else 'Non renseigné'
        else:
            user_info['Fournisseur'] = 'Non renseigné'
        
        # Traitement sécurisé de medium
        if pd.notna(user_data['medium']):
            medium = str(user_data['medium'])
            user_info['Source'] = medium if medium != '(not set)' else 'Non renseigné'
        else:
            user_info['Source'] = 'Non renseigné'
        
        for key, value in user_info.items():
            st.write(f"**{key}:** {value}")
        
        # Date de dernière visite
        st.write(f"**Dernière visite:** {user_data['full_date']}")
        
        # Prochaine visite prévue
        st.write(f"**Prochaine visite prévue:** {user_data['next_full_date']}")
        
        # Centres d'intérêt basés sur le cluster
        if user_cluster in cluster_profiles:
            st.subheader("Centres d'intérêt probables")
            for interest in cluster_profiles[user_cluster]['interests']:
                st.write(f"- {interest}")
    
    with col2:
        # Types de contenu préférés
        if user_cluster in cluster_profiles:
            st.subheader("Types de contenu préférés")
            for content_type in cluster_profiles[user_cluster]['content_types']:
                st.write(f"- {content_type}")
        
        # Emplacement pour future fonctionnalité : Analyse de l'activité par jour
        st.subheader("Analyse temporelle")
        st.markdown("📅 **Analyse de l'activité par jour de la semaine**")
        
        # Vérifier si les données existent
        if 'dayofweek' in df.columns and pd.notna(user_data['dayofweek']):
            # Utiliser les données réelles
            day_data = user_data['dayofweek']
            st.write(f"Jour de dernière activité: **{day_data}**")
            
            # Afficher un message pour la future fonctionnalité
            st.info("🔜 **Bientôt disponible :** Visualisation complète de l'activité par jour de la semaine")
        else:
            st.info("🔜 **Bientôt disponible :** Analyse de l'activité par jour de la semaine")
        
        # Emplacement pour future fonctionnalité : Analyse des heures de connexion
        st.markdown("🕒 **Analyse des heures de connexion préférées**")
        
        # Vérifier si les données existent
        if 'hour' in df.columns and pd.notna(user_data['hour']):
            # Utiliser les données réelles
            hour_data = user_data['hour']
            st.write(f"Heure de dernière activité: **{hour_data}h**")
            
            # Afficher un message pour la future fonctionnalité
            st.info("🔜 **Bientôt disponible :** Visualisation complète des heures de connexion préférées")
        else:
            st.info("🔜 **Bientôt disponible :** Analyse des heures de connexion préférées")
        
        # Emplacement pour future fonctionnalité : Badges et récompenses
        st.subheader("Badges et récompenses")
        st.info("🔜 **Bientôt disponible :** Système de badges et récompenses basé sur l'activité et les contributions")
        
        # Afficher un exemple de badge à venir
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("🥉 **Bronze**")
        with col2:
            st.markdown("🥈 **Argent**")
        with col3:
            st.markdown("🥇 **Or**")

with tabs[2]:  # Prédictions
    col1, col2 = st.columns(2)
    
    with col1:
        # Prédiction avec le modèle
        st.subheader("🔮 Prédiction de désengagement")
        
        if model is not None:
            # Préparation des données pour la prédiction
            features = [
                'day', 'month', 'weekofyear', 'is_weekend',
                'quarter', 'semester', 'year', 'executive_weighted_score'
            ]
            
            # Vérification que toutes les features sont présentes
            available_features = [f for f in features if f in df.columns]
            
            if len(available_features) == len(features):
                # Création d'un DataFrame avec les données de l'utilisateur
                user_df = pd.DataFrame([user_data[features]])
                
                # Prédiction
                try:
                    prediction_proba = model.predict_proba(user_df)[:, 1][0]
                    prediction = model.predict(user_df)[0]
                    
                    # Affichage de la prédiction
                    st.write(f"**Probabilité de désengagement:** {prediction_proba:.2%}")
                    
                    # Jauge de risque
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prediction_proba * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Risque de désengagement"},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkred"},
                            'steps': [
                                {'range': [0, 30], 'color': "green"},
                                {'range': [30, 70], 'color': "orange"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': prediction_proba * 100
                            }
                        }
                    ))
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Facteurs de risque
                    st.subheader("Facteurs de risque")
                    
                    risk_factors = []
                    if user_data['recency_days'] > 30:
                        risk_factors.append(f"⚠️ Inactivité prolongée ({user_data['recency_days']} jours)")
                    
                    if 'bounce_rate' in user_data and user_data['bounce_rate'] > 0.7:
                        risk_factors.append(f"⚠️ Taux de rebond élevé ({user_data['bounce_rate']:.2f})")
                    
                    if user_data['engagement_trend'] < 0:
                        risk_factors.append(f"⚠️ Tendance d'engagement négative ({user_data['engagement_trend']:.2f})")
                    
                    if user_data['nb_sessions'] < 3:
                        risk_factors.append(f"⚠️ Nombre de sessions faible ({user_data['nb_sessions']})")
                    
                    if not risk_factors:
                        risk_factors.append("✅ Aucun facteur de risque majeur identifié")
                    
                    for factor in risk_factors:
                        st.write(factor)
                    
                except Exception as e:
                    st.error(f"Erreur lors de la prédiction: {e}")
            else:
                st.warning("Certaines caractéristiques nécessaires pour la prédiction ne sont pas disponibles dans les données.")
        else:
            st.info("Le modèle de prédiction n'est pas disponible. Impossible de générer des prédictions.")
    
    with col2:
        # Prédiction de la prochaine visite
        st.subheader("📅 Prédiction de la prochaine visite")
        
        # Calculer la date estimée de la prochaine visite
        last_visit = pd.to_datetime(user_data['full_date'])
        avg_days = max(1, float(user_data['avg_days_between_sessions']) if pd.notna(user_data['avg_days_between_sessions']) else 7)
        
        # Ajuster la prédiction en fonction du risque de désengagement
        if user_data['will_disengage_30d'] == 1:
            avg_days *= 1.5  # Augmenter l'intervalle si risque de désengagement
        
        next_visit_prediction = last_visit + pd.Timedelta(days=avg_days)
        
        # Afficher la prédiction
        days_until_next = (next_visit_prediction - datetime.now()).days
        
        if days_until_next < 0:
            st.warning(f"⚠️ **Retard de visite:** L'utilisateur aurait dû revenir il y a {abs(days_until_next)} jours")
        else:
            st.info(f"📆 **Prochaine visite estimée:** {next_visit_prediction.strftime('%d/%m/%Y')} (dans {days_until_next} jours)")
        
        # Prédiction des intérêts
        st.subheader("🔍 Centres d'intérêt prédits")
        
        if user_cluster in cluster_profiles:
            # Afficher les recommandations de contenu basées sur le cluster
            for recommendation in cluster_profiles[user_cluster]['recommendations']:
                st.write(f"- {recommendation}")
        
        # Emplacement pour future fonctionnalité : Prédiction de valeur utilisateur
        st.subheader("💰 Valeur utilisateur")
        st.info("🔜 **Bientôt disponible :** Estimation de la valeur utilisateur et du potentiel de croissance")
        
        # Afficher un aperçu de la future fonctionnalité
        st.markdown("**Métriques à venir :**")
        st.markdown("- Valeur actuelle estimée")
        st.markdown("- Potentiel de croissance")
        st.markdown("- Probabilité de conversion premium")
        st.markdown("- Impact sur la communauté")

with tabs[3]:  # Historique
    # Historique des interactions
    st.subheader("📅 Historique des interactions")
    
    # Création d'un historique basé sur les données disponibles
    nb_sessions = int(user_data['nb_sessions'])
    
    if nb_sessions > 0:
        # Afficher les informations de session disponibles
        st.write(f"**Nombre total de sessions :** {nb_sessions}")
        st.write(f"**Dernière visite :** {user_data['full_date']}")
        st.write(f"**Temps moyen entre les sessions :** {user_data['avg_days_between_sessions']:.1f} jours")
        
        # Emplacement pour future fonctionnalité : Historique détaillé
        st.info("🔜 **Bientôt disponible :** Historique détaillé des interactions avec dates, actions et durées")
        
        # Aperçu du futur tableau d'historique
        st.markdown("**Aperçu du futur historique détaillé :**")
        preview_df = pd.DataFrame({
            'Date': ["JJ/MM/AAAA", "JJ/MM/AAAA", "JJ/MM/AAAA"],
            'Action': ["Type d'action", "Type d'action", "Type d'action"],
            'Page': ["Section visitée", "Section visitée", "Section visitée"],
            'Durée': ["XX min", "XX min", "XX min"]
        })
        st.dataframe(preview_df, use_container_width=True)
    else:
        st.info("Aucune session enregistrée pour cet utilisateur.")
    
    # Emplacement pour future fonctionnalité : Évolution de l'engagement
    st.subheader("📈 Évolution de l'engagement")
    st.info("🔜 **Bientôt disponible :** Graphique d'évolution de l'engagement au fil du temps")
    
    # Afficher un aperçu du futur graphique
    st.markdown("**Métriques qui seront suivies :**")
    st.markdown("- Score d'engagement mensuel")
    st.markdown("- Nombre de sessions")
    st.markdown("- Temps passé sur la plateforme")
    st.markdown("- Contributions à la communauté")
    
    # Emplacement pour future fonctionnalité : Parcours utilisateur
    st.subheader("🛤️ Parcours utilisateur")
    st.info("🔜 **Bientôt disponible :** Visualisation du parcours utilisateur à travers la plateforme")

with tabs[4]:  # Actions recommandées
    col1, col2 = st.columns(2)
    
    with col1:
        # Recommandations personnalisées
        st.subheader("🎯 Recommandations personnalisées")
        
        # Définir des recommandations en fonction du profil
        if user_data['is_disengaged'] == 1:
            st.error("⚠️ **Utilisateur désengagé**")
            
            # Recommandations spécifiques pour utilisateur désengagé
            st.markdown("### Plan de réactivation recommandé")
            
            st.markdown("**1. Campagne de réactivation ciblée**")
            st.markdown("- Email personnalisé avec contenu adapté à son segment")
            st.markdown("- Offre spéciale d'accès à du contenu premium")
            st.markdown("- Rappel des derniers contenus consultés")
            
            if user_cluster in cluster_profiles:
                st.markdown(f"**2. Contenu recommandé pour ce segment**")
                for i, recommendation in enumerate(cluster_profiles[user_cluster]['recommendations']):
                    st.markdown(f"- {recommendation}")
            
            st.markdown("**3. Incitatifs à l'engagement**")
            st.markdown("- Accès temporaire à des fonctionnalités premium")
            st.markdown("- Invitation à un webinaire exclusif")
            st.markdown("- Sondage pour comprendre ses besoins spécifiques")
            
            if st.button("📧 Envoyer email de réactivation", key="reactivation_email"):
                if user_cluster in cluster_profiles:
                    template = cluster_profiles[user_cluster]['email_template']
                    st.success(f"✅ Email de réactivation programmé avec le modèle: '{template['subject']}'")
                    st.code(template['content'], language="text")
                else:
                    st.success("✅ Email de réactivation programmé!")
        
        elif user_data['will_disengage_30d'] == 1:
            st.warning("⚠️ **Risque élevé de désengagement**")
            
            # Recommandations spécifiques pour risque de désengagement
            st.markdown("### Programme de fidélisation préventif")
            
            st.markdown("**1. Actions de rétention immédiates**")
            st.markdown("- Email personnalisé avec contenu à forte valeur ajoutée")
            st.markdown("- Notification push pour signaler de nouveaux contenus pertinents")
            st.markdown("- Relance avec offre spéciale limitée dans le temps")
            
            if user_cluster in cluster_profiles:
                st.markdown(f"**2. Contenu recommandé pour ce segment**")
                for i, recommendation in enumerate(cluster_profiles[user_cluster]['recommendations']):
                    st.markdown(f"- {recommendation}")
            
            st.markdown("**3. Stratégie d'engagement**")
            st.markdown("- Parcours personnalisé basé sur ses centres d'intérêt")
            st.markdown("- Mini-challenge adapté à son niveau")
            st.markdown("- Invitation à rejoindre un groupe de discussion thématique")
            
            if st.button("🎁 Envoyer offre de fidélisation", key="loyalty_offer"):
                if user_cluster in cluster_profiles:
                    template = cluster_profiles[user_cluster]['email_template']
                    st.success(f"✅ Offre de fidélisation programmée avec le modèle: '{template['subject']}'")
                    st.code(template['content'], language="text")
                else:
                    st.success("✅ Offre de fidélisation programmée!")
        
        else:
            st.success("✅ **Utilisateur engagé**")
            
            # Recommandations spécifiques pour utilisateur engagé
            st.markdown("### Programme d'approfondissement")
            
            st.markdown("**1. Opportunités d'engagement avancé**")
            st.markdown("- Invitation à contribuer à la communauté")
            st.markdown("- Accès à des contenus premium exclusifs")
            st.markdown("- Participation à des Data Challenges")
            
            if user_cluster in cluster_profiles:
                st.markdown(f"**2. Contenu recommandé pour ce segment**")
                for i, recommendation in enumerate(cluster_profiles[user_cluster]['recommendations']):
                    st.markdown(f"- {recommendation}")
            
            st.markdown("**3. Valorisation de l'expertise**")
            st.markdown("- Programme de reconnaissance (badges, statuts)")
            st.markdown("- Opportunités de partage de connaissances")
            st.markdown("- Invitation à des événements exclusifs")
            
            if st.button("⭐ Proposer offre premium", key="premium_offer"):
                if user_cluster in cluster_profiles:
                    template = cluster_profiles[user_cluster]['email_template']
                    st.success(f"✅ Proposition d'offre premium programmée avec le modèle: '{template['subject']}'")
                    st.code(template['content'], language="text")
                else:
                    st.success("✅ Proposition d'offre premium programmée!")
    
    with col2:
        # Plan d'action personnalisé
        st.subheader("📋 Plan d'action personnalisé")
        
        # Créer un plan d'action basé sur le profil utilisateur
        today = datetime.now()
        
        # Déterminer le type de plan en fonction du statut
        if user_data['is_disengaged'] == 1:
            plan_type = "Réactivation"
            actions = [
                {"date": today + timedelta(days=0), "action": "Envoyer email de réactivation personnalisé", "statut": "À faire"},
                {"date": today + timedelta(days=3), "action": "Relance SMS si pas de réponse", "statut": "Planifié"},
                {"date": today + timedelta(days=7), "action": "Offre spéciale d'accès premium temporaire", "statut": "Planifié"},
                {"date": today + timedelta(days=14), "action": "Sondage pour comprendre les raisons du désengagement", "statut": "Planifié"},
                {"date": today + timedelta(days=21), "action": "Analyse des résultats et ajustement de la stratégie", "statut": "Planifié"}
            ]
        elif user_data['will_disengage_30d'] == 1:
            plan_type = "Rétention"
            actions = [
                {"date": today + timedelta(days=0), "action": "Envoyer email avec contenu personnalisé à forte valeur", "statut": "À faire"},
                {"date": today + timedelta(days=2), "action": "Notification push avec nouveaux contenus pertinents", "statut": "Planifié"},
                {"date": today + timedelta(days=5), "action": "Invitation à un mini-challenge adapté", "statut": "Planifié"},
                {"date": today + timedelta(days=10), "action": "Proposer un parcours thématique personnalisé", "statut": "Planifié"},
                {"date": today + timedelta(days=15), "action": "Évaluation de l'engagement et ajustement", "statut": "Planifié"}
            ]
        else:
            plan_type = "Développement"
            actions = [
                {"date": today + timedelta(days=0), "action": "Proposer du contenu premium exclusif", "statut": "À faire"},
                {"date": today + timedelta(days=3), "action": "Invitation à contribuer à la communauté", "statut": "Planifié"},
                {"date": today + timedelta(days=7), "action": "Proposition de participation à un Data Challenge", "statut": "Planifié"},
                {"date": today + timedelta(days=14), "action": "Invitation à un événement exclusif", "statut": "Planifié"},
                {"date": today + timedelta(days=21), "action": "Programme de reconnaissance et valorisation", "statut": "Planifié"}
            ]
        
        # Créer un DataFrame pour le plan d'action
        plan_df = pd.DataFrame(actions)
        plan_df['Date'] = plan_df['date'].dt.strftime('%d/%m/%Y')
        plan_df = plan_df[['Date', 'action', 'statut']]
        plan_df.columns = ['Date', 'Action', 'Statut']
        
        # Afficher le plan d'action
        st.markdown(f"**Plan de {plan_type}**")
        st.dataframe(plan_df, use_container_width=True)
        
        # Bouton pour générer un plan détaillé
        if st.button("📝 Générer plan d'action détaillé", key="generate_plan"):
            plan_content = f"# Plan d'action détaillé - {plan_type}\n\n"
            plan_content += f"Utilisateur: {user_data['visitor_id']}\n"
            plan_content += f"Segment: {user_cluster} - {cluster_profiles.get(user_cluster, {}).get('title', 'Non défini')}\n"
            plan_content += f"Date de génération: {today.strftime('%d/%m/%Y')}\n\n"
            
            plan_content += "## Actions planifiées\n\n"
            for action in actions:
                plan_content += f"### {action['date'].strftime('%d/%m/%Y')} - {action['action']}\n"
                plan_content += f"Statut: {action['statut']}\n\n"
                
                # Ajouter des détails selon le type d'action
                if "email" in action['action'].lower():
                    if user_cluster in cluster_profiles:
                        template = cluster_profiles[user_cluster]['email_template']
                        plan_content += f"**Modèle d'email:**\n"
                        plan_content += f"Objet: {template['subject']}\n"
                        plan_content += f"Contenu:\n{template['content']}\n\n"
                
                elif "contenu" in action['action'].lower() and user_cluster in cluster_profiles:
                    plan_content += "**Contenus recommandés:**\n"
                    for rec in cluster_profiles[user_cluster]['recommendations']:
                        plan_content += f"- {rec}\n"
                    plan_content += "\n"
            
            plan_content += "## Métriques à surveiller\n\n"
            plan_content += "- Taux d'ouverture des emails\n"
            plan_content += "- Taux de clics sur les contenus recommandés\n"
            plan_content += "- Nombre de sessions après chaque action\n"
            plan_content += "- Évolution du score d'engagement\n"
            plan_content += "- Temps passé sur la plateforme\n\n"
            
            plan_content += "## Objectifs\n\n"
            if user_data['is_disengaged'] == 1:
                plan_content += "- Réactiver l'utilisateur avec au moins 2 sessions dans les 30 prochains jours\n"
                plan_content += "- Augmenter le score d'engagement de 50% minimum\n"
                plan_content += "- Obtenir au moins une interaction significative (commentaire, téléchargement, etc.)\n"
            elif user_data['will_disengage_30d'] == 1:
                plan_content += "- Maintenir l'activité avec au moins 3 sessions dans les 30 prochains jours\n"
                plan_content += "- Augmenter le score d'engagement de 20% minimum\n"
                plan_content += "- Encourager la participation à au moins un événement ou challenge\n"
            else:
                plan_content += "- Approfondir l'engagement avec au moins 5 sessions dans les 30 prochains jours\n"
                plan_content += "- Obtenir au moins une contribution à la communauté\n"
                plan_content += "- Convertir vers une utilisation premium ou un statut d'ambassadeur\n"
            
            st.download_button(
                label="⬇️ Télécharger le plan d'action",
                data=plan_content,
                file_name=f"plan_action_{user_data['visitor_id']}.md",
                mime="text/markdown"
            )
        
        # Modèles de communication
        st.subheader("💬 Modèles de communication")
        
        if user_cluster in cluster_profiles:
            template = cluster_profiles[user_cluster]['email_template']
            
            comm_type = st.selectbox(
                "Type de communication",
                ["Email", "SMS", "Notification Push"]
            )
            
            if comm_type == "Email":
                st.markdown(f"**Objet suggéré:** {template['subject']}")
                st.text_area("Contenu de l'email", template['content'], height=200)
            elif comm_type == "SMS":
                sms_content = f"Management & Data Science: {template['subject'].replace('🔍 ', '').replace('🔄 ', '').replace('🎁 ', '').replace('🏆 ', '').replace('📊 ', '').replace('🌟 ', '').replace('👑 ', '').replace('🔑 ', '').replace('🌠 ', '')}"
                st.text_area("Contenu du SMS", sms_content[:160], height=100)
                st.write(f"Caractères: {len(sms_content[:160])}/160")
            else:  # Notification Push
                notif_title = template['subject'].replace('🔍 ', '').replace('🔄 ', '').replace('🎁 ', '').replace('🏆 ', '').replace('📊 ', '').replace('🌟 ', '').replace('👑 ', '').replace('🔑 ', '').replace('🌠 ', '')
                st.text_input("Titre de la notification", notif_title[:50])

with tabs[5]:  # Fonctionnalités à venir
    st.header("🚀 Fonctionnalités à venir - Mai 2025")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧠 IA Prédictive Avancée")
        st.markdown("""
        **Disponible prochainement :** Notre nouveau modèle d'IA prédictive vous permettra d'anticiper avec précision le comportement des utilisateurs.
        
        **Fonctionnalités incluses :**
        - Prédiction multi-horizon (7j, 30j, 90j)
        - Identification des facteurs d'influence personnalisés
        - Recommandations d'actions préventives automatisées
        - Segmentation dynamique basée sur les comportements prédits
        
        **Bénéfices :** Réduisez le désengagement de 35% et augmentez la rétention de vos utilisateurs les plus précieux.
        """)
        
        st.image("https://via.placeholder.com/400x200?text=IA+Predictive+Preview", use_column_width=True)
        
        st.subheader("🔄 Parcours Utilisateur Interactif")
        st.markdown("""
        **En développement :** Visualisez le parcours complet de vos utilisateurs à travers votre plateforme avec notre nouvelle interface interactive.
        
        **Fonctionnalités incluses :**
        - Cartographie visuelle du parcours utilisateur
        - Identification des points de friction
        - Analyse des chemins de conversion
        - Comparaison avec les parcours optimaux
        
        **Bénéfices :** Optimisez votre UX et augmentez les taux de conversion de 25%.
        """)
    
    with col2:
        st.subheader("🎯 Recommandations Personnalisées par IA")
        st.markdown("""
        **En phase de test :** Notre système de recommandation basé sur l'IA analysera les comportements et préférences pour suggérer le contenu parfaitement adapté à chaque utilisateur.
        
        **Fonctionnalités incluses :**
        - Recommandations de contenu ultra-personnalisées
        - Suggestions de connexions avec d'autres membres
        - Identification des opportunités de contribution
        - Adaptation dynamique aux changements d'intérêts
        
        **Bénéfices :** Augmentez l'engagement de 40% et le temps passé sur la plateforme de 65%.
        """)
        
        st.image("https://via.placeholder.com/400x200?text=Recommandations+IA+Preview", use_column_width=True)
        
        st.subheader("🏆 Système de Gamification Avancé")
        st.markdown("""
        **Lancement prévu :** Notre nouveau système de gamification transformera l'expérience utilisateur en un parcours engageant et gratifiant.
        
        **Fonctionnalités incluses :**
        - Badges et récompenses personnalisés
        - Niveaux d'expertise progressifs
        - Défis adaptés au profil de chaque utilisateur
        - Tableaux de classement par domaine d'expertise
        
        **Bénéfices :** Stimulez les contributions de qualité et renforcez le sentiment d'appartenance à la communauté.
        """)
    
    # Timeline de déploiement
    st.subheader("📅 Calendrier de déploiement")
    
    timeline_data = pd.DataFrame([
        {"Phase": "IA Prédictive Avancée", "Début": "Mai 2025", "Fin": "Juin 2025", "Statut": "En développement"},
        {"Phase": "Parcours Utilisateur Interactif", "Début": "Juin 2025", "Fin": "Juillet 2025", "Statut": "Planifié"},
        {"Phase": "Recommandations Personnalisées par IA", "Début": "Juillet 2025", "Fin": "Août 2025", "Statut": "En phase de test"},
        {"Phase": "Système de Gamification Avancé", "Début": "Août 2025", "Fin": "Septembre 2025", "Statut": "Planifié"}
    ])
    
    st.dataframe(timeline_data, use_container_width=True)
    
    # Appel à l'action
    st.success("🔔 **Inscrivez-vous au programme bêta-testeurs pour essayer ces fonctionnalités en avant-première !**")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.button("✉️ S'inscrire au programme bêta", use_container_width=True)
    with col2:
        st.text_input("Email pour recevoir les mises à jour", placeholder="votre.email@exemple.com")

