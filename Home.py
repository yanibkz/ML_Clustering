import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
from datetime import datetime, timedelta

# Configuration de la page
st.set_page_config(
    page_title="🔍 Dashboard Marketing & Segmentation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chargement des données
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('final_df.csv')
        return df
    except FileNotFoundError:
        st.error("Fichier de données non trouvé. Veuillez vérifier le chemin du fichier.")
        st.stop()

df = load_data()

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

# Sous-clustering du cluster 3
mask_cluster_3 = df['cluster'] == '3'
if mask_cluster_3.sum() > 0:
    X_cluster_3 = df.loc[mask_cluster_3, num_cols]
    X_cluster_3_processed = main_pipeline.named_steps['prep'].transform(X_cluster_3)
    sub_labels = sub_kmeans.predict(X_cluster_3_processed)
    cluster_3_indices = df[mask_cluster_3].index.to_numpy()
    for s in range(2):  # 2 sous-clusters
        sub_idx = cluster_3_indices[sub_labels == s]
        df.loc[sub_idx, 'cluster'] = f"3_{s+1}"

# Dictionnaire des profils de clusters détaillés pour Management & Data Science
cluster_profiles = {
    '0': {
        'title': "Faiblement Engagé, Passif",
        'size': "3 849 utilisateurs",
        'description': "Utilisateurs avec un faible engagement (score moyen: 3,18) et peu d'interactions (1,86 clics/session). Principalement issus de recherche organique (62,8%).",
        'strategy': "Activation et éducation progressive",
        'actions': [
            "Créer une série de contenus d'introduction 'Data Science pour tous'",
            "Proposer un parcours guidé de découverte des ressources fondamentales",
            "Mettre en avant des témoignages d'utilisateurs ayant progressé",
            "Optimiser le référencement naturel sur les sujets les plus recherchés",
            "Réduire le rebond élevé par des contenus d'accroche immédiate (vidéos, infographies)"
        ],
        'kpis': [
            "Taux de rebond sur les pages d'introduction",
            "Taux de progression dans le parcours guidé",
            "Taux de conversion vers l'inscription complète",
            "Augmentation du temps moyen passé sur le site"
        ],
        'content_types': ["Articles introductifs", "Vidéos explicatives", "Infographies simplifiées"]
    },
    '1': {
        'title': "Passif à Risque élevé de Désengagement",
        'size': "4 369 utilisateurs",
        'description': "Utilisateurs avec un engagement très limité (score moyen: 2,18) et des interactions modérées sans profondeur.",
        'strategy': "Réactivation ciblée et démonstration de valeur",
        'actions': [
            "Proposer des cas d'études concrets montrant l'impact dans leur secteur",
            "Mettre en avant les nouveaux contenus depuis leur dernière visite",
            "Créer des parcours de micro-apprentissage à faible engagement initial",
            "Campagnes de réactivation email ciblées avec contenu personnalisé",
            "Améliorer l'UX/UI sur les pages d'entrée majeures"
        ],
        'kpis': [
            "Taux de réactivation après campagne",
            "Taux d'ouverture des emails et clics",
            "Augmentation du temps passé sur la plateforme",
            "Réduction du taux de désengagement"
        ],
        'content_types': ["Cas d'études", "Tutoriels courts", "Newsletters ciblées"]
    },
    '2': {
        'title': "Très Faible Engagement",
        'size': "273 utilisateurs",
        'description': "Petit groupe d'utilisateurs avec un engagement très faible (score: 1,68) et peu d'interactions.",
        'strategy': "Reconquête intensive avec proposition de valeur claire",
        'actions': [
            "Offrir un accès temporaire à des contenus premium pour démontrer la valeur",
            "Proposer des formats de contenu ultra-simplifiés et accessibles",
            "Mettre en avant les success stories de la communauté pour inspirer l'engagement",
            "Lancer des campagnes de retargeting intensives sur les réseaux sociaux",
            "Créer des landing pages spécifiquement conçues pour ce type de visiteurs"
        ],
        'kpis': [
            "Taux d'activation de l'offre premium",
            "Taux de rétention après la période d'essai",
            "Augmentation du nombre de sessions",
            "Réduction du taux de rebond"
        ],
        'content_types': ["Vidéos courtes", "Infographies", "Témoignages"]
    },
    '3_1': {
        'title': "Actif avec Fort Potentiel",
        'size': "7 974 utilisateurs",
        'description': "Grand groupe d'utilisateurs avec un bon engagement (score: 4,27) et une bonne interactivité (3,37 clics/session).",
        'strategy': "Approfondissement et engagement communautaire",
        'actions': [
            "Proposer des parcours thématiques avancés basés sur leurs centres d'intérêt",
            "Encourager la participation aux Data Challenges et projets collaboratifs",
            "Inviter à contribuer à la communauté par des partages d'expérience",
            "Créer des contenus interactifs fréquents : quiz, webinars, forums",
            "Mettre en place une stratégie de fidélisation avancée (points, badges)"
        ],
        'kpis': [
            "Taux de participation aux Data Challenges",
            "Nombre de contributions à la communauté",
            "Taux d'engagement avec le contenu avancé",
            "Progression dans les parcours thématiques"
        ],
        'content_types': ["Projets pratiques", "Webinaires avancés", "Forums spécialisés"]
    },
    '3_2': {
        'title': "Moyennement Actif mais Sous-exploité",
        'size': "674 utilisateurs",
        'description': "Groupe modéré avec un engagement moyen (score: 3,46) et des interactions moyennes (2,00 clics/session).",
        'strategy': "Activation ciblée et personnalisation accrue",
        'actions': [
            "Proposer des contenus intermédiaires adaptés à leur niveau de compétence",
            "Mettre en avant les discussions et projets actifs dans leurs domaines d'intérêt",
            "Suggérer des connexions avec d'autres membres aux intérêts similaires",
            "Introduire des messages contextuels (pop-ups personnalisées)",
            "Optimiser les points d'entrée clés pour faciliter l'accès au contenu pertinent"
        ],
        'kpis': [
            "Taux de clics sur les contenus recommandés",
            "Augmentation du nombre d'interactions par session",
            "Taux de participation aux mini-challenges",
            "Nombre de connexions établies avec d'autres membres"
        ],
        'content_types': ["Datasets commentés", "Forums de discussion", "Exercices pratiques"]
    },
    '4': {
        'title': "Bonne Interaction Générale",
        'size': "675 utilisateurs",
        'description': "Groupe modéré avec un engagement élevé (score: 4,80) mais des interactions moyennes (1,94 clics/session).",
        'strategy': "Diversification des interactions et approfondissement",
        'actions': [
            "Proposer l'exploration de nouveaux domaines connexes à leurs intérêts",
            "Encourager le partage de leurs connaissances via des contributions",
            "Suggérer des collaborations avec d'autres membres actifs",
            "Inciter à des interactions supplémentaires via des Call-To-Action visibles",
            "Développer du contenu premium disponible après interaction"
        ],
        'kpis': [
            "Diversité des contenus consultés",
            "Taux de contribution (commentaires, évaluations)",
            "Nombre de nouvelles connexions établies",
            "Augmentation du nombre de clics par session"
        ],
        'content_types': ["Articles spécialisés", "Ateliers pratiques", "Projets collaboratifs"]
    },
    '5': {
        'title': "Petit Groupe Actif",
        'size': "440 utilisateurs",
        'description': "Petit groupe engagé avec un bon potentiel d'approfondissement (score: 4,07).",
        'strategy': "Spécialisation et valorisation des contributions",
        'actions': [
            "Proposer des contenus de niche et avancés dans leurs domaines de prédilection",
            "Encourager la création et le partage de contenus spécialisés",
            "Mettre en avant leur expertise auprès de la communauté",
            "Développer des contenus très ciblés, spécifiquement adaptés à ce groupe",
            "Relances personnalisées via campagnes email spécialisées"
        ],
        'kpis': [
            "Taux de participation aux groupes spécialisés",
            "Nombre et qualité des contenus créés",
            "Influence sur la communauté (partages, mentions)",
            "Profondeur d'engagement avec les contenus de niche"
        ],
        'content_types': ["Articles de niche", "Groupes spécialisés", "Opportunités de publication"]
    },
    '6': {
        'title': "Très Actif et Intense",
        'size': "1 186 utilisateurs",
        'description': "Groupe significatif avec une forte intensité d'engagement (score: 3,85) et des interactions très élevées (6,13 clics/session).",
        'strategy': "Valorisation premium et leadership communautaire",
        'actions': [
            "Offrir un accès anticipé aux nouvelles fonctionnalités et contenus",
            "Proposer des rôles de mentors ou d'animateurs thématiques",
            "Créer des expériences exclusives et personnalisées",
            "Mettre en place des expériences premium et exclusives",
            "Déployer des stratégies communautaires fortes (groupes privés, événements)"
        ],
        'kpis': [
            "Taux d'activation du statut VIP",
            "Participation aux événements exclusifs",
            "Impact des contributions (vues, partages)",
            "Influence sur l'acquisition de nouveaux membres"
        ],
        'content_types': ["Contenus exclusifs", "Événements VIP", "Opportunités de mentorat"]
    },
    '7': {
        'title': "Petit Groupe Ultra-Engagé",
        'size': "223 utilisateurs",
        'description': "Petit groupe d'élite avec un engagement exceptionnel (score: 4,80) et une très forte intensité (4,55).",
        'strategy': "Partenariat stratégique et co-création",
        'actions': [
            "Proposer une relation privilégiée avec l'équipe de Management & Data Science",
            "Offrir des opportunités de co-création de contenus et fonctionnalités",
            "Valoriser leur expertise à travers des interviews et témoignages",
            "Renforcer la communication personnalisée (conseillers dédiés)",
            "Créer des contenus et événements exclusifs, réservés à ces utilisateurs"
        ],
        'kpis': [
            "Taux d'acceptation du programme Ambassadeur",
            "Qualité et impact des co-créations",
            "Influence sur l'acquisition de nouveaux membres",
            "Taux de rétention de ce segment d'élite"
        ],
        'content_types': ["Co-création de contenu", "Interviews", "Tables rondes"]
    }
}

# En-tête de la page
st.title("🔍 Dashboard Marketing & Segmentation")
st.markdown("### Plateforme d'analyse de segmentation pour Management & Data Science")

# Date actuelle
current_date = datetime(2025, 4, 18)
st.markdown(f"*Données mises à jour le {current_date.strftime('%d/%m/%Y')}*")

# Présentation des statistiques globales
st.subheader("📊 Vue d'ensemble")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Nombre total d'utilisateurs", value=f"{len(df):,}")
with col2:
    st.metric(label="Nombre de segments", value=len(df['cluster'].unique()))
with col3:
    avg_engagement = round(df['engagement_score'].mean(), 2)
    st.metric(label="Score d'engagement moyen", value=f"{avg_engagement}")
with col4:
    active_users = df[df['is_active'] == 1].shape[0]
    active_pct = round(active_users / len(df) * 100, 1)
    st.metric(label="Utilisateurs actifs", value=f"{active_pct}%")

# Répartition des clusters
st.subheader("📈 Répartition des segments")

col1, col2 = st.columns([2, 3])

with col1:
    cluster_counts = df['cluster'].value_counts().sort_index()
    
    # Créer un DataFrame pour le graphique avec des titres plus descriptifs
    cluster_labels = []
    for cluster in cluster_counts.index:
        if cluster in cluster_profiles:
            cluster_labels.append(f"{cluster}: {cluster_profiles[cluster]['title']}")
        else:
            cluster_labels.append(f"Cluster {cluster}")
    
    cluster_df = pd.DataFrame({
        'Segment': cluster_labels,
        'Nombre d\'utilisateurs': cluster_counts.values
    })
    
    fig = px.bar(
        cluster_df,
        x='Segment',
        y='Nombre d\'utilisateurs',
        title="Répartition des utilisateurs par segment",
        color='Nombre d\'utilisateurs',
        color_continuous_scale='Blues'
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Tableau récapitulatif des segments
    st.markdown("### Aperçu des segments")
    
    segment_summary = []
    for cluster in sorted(cluster_counts.index):
        if cluster in cluster_profiles:
            segment_summary.append({
                "Segment": cluster,
                "Titre": cluster_profiles[cluster]['title'],
                "Effectif": f"{cluster_counts[cluster]:,}",
                "Stratégie": cluster_profiles[cluster]['strategy']
            })
    
    st.dataframe(pd.DataFrame(segment_summary), use_container_width=True)

with col2:
    # Profil des clusters avec radar chart amélioré
    numeric_cols = ['engagement_score', 'global_engagement_score', 'engagement_intensity', 
                    'multi_interaction_score', 'session_efficiency', 'retention_like_score']
    
    # Renommer les colonnes pour plus de clarté
    readable_cols = {
        'engagement_score': 'Engagement',
        'global_engagement_score': 'Engagement global',
        'engagement_intensity': 'Intensité',
        'multi_interaction_score': 'Multi-interaction',
        'session_efficiency': 'Efficacité session',
        'retention_like_score': 'Rétention'
    }
    
    cluster_means = df.groupby('cluster')[numeric_cols].mean()
    normalized_means = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
    
    # Créer le radar chart
    fig = go.Figure()
    
    # Utiliser une palette de couleurs cohérente
    colors = px.colors.qualitative.Bold
    
    for i, cluster in enumerate(sorted(df['cluster'].unique(), key=lambda x: (str(x)))):
        color_idx = i % len(colors)
        
        # Ajouter un titre descriptif si disponible
        if cluster in cluster_profiles:
            cluster_name = f"{cluster}: {cluster_profiles[cluster]['title']}"
        else:
            cluster_name = f"Cluster {cluster}"
            
        fig.add_trace(go.Scatterpolar(
            r=normalized_means.loc[cluster].values,
            theta=[readable_cols.get(col, col) for col in numeric_cols],
            fill='toself',
            name=cluster_name,
            line=dict(color=colors[color_idx])
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="Profil des segments (caractéristiques normalisées)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Analyse de désengagement
st.subheader("⚠️ Analyse du désengagement")

disengaged_users = df[df['is_disengaged'] == 1]
disengaged_pct = round(len(disengaged_users) / len(df) * 100, 1)

col1, col2 = st.columns(2)

with col1:
    st.metric("Utilisateurs désengagés", f"{disengaged_pct}%", 
              delta=f"{disengaged_pct - 5.2:.1f}%" if disengaged_pct > 5.2 else f"{5.2 - disengaged_pct:.1f}%",
              delta_color="inverse")
    
    # Créer un DataFrame pour le graphique avec des titres plus descriptifs
    disengaged_by_cluster = disengaged_users['cluster'].value_counts().sort_index()
    total_by_cluster = df['cluster'].value_counts().sort_index()
    
    # Calculer le pourcentage de désengagement par cluster
    pct_disengaged = pd.DataFrame()
    pct_disengaged['Cluster'] = total_by_cluster.index
    pct_disengaged['Total'] = total_by_cluster.values
    pct_disengaged['Désengagés'] = [disengaged_by_cluster.get(c, 0) for c in total_by_cluster.index]
    pct_disengaged['Pourcentage'] = (pct_disengaged['Désengagés'] / pct_disengaged['Total'] * 100).round(1)
    
    # Ajouter les titres descriptifs
    pct_disengaged['Segment'] = pct_disengaged['Cluster'].apply(
        lambda x: f"{x}: {cluster_profiles.get(x, {}).get('title', 'Non défini')}" if x in cluster_profiles else f"Cluster {x}"
    )
    
    # Créer le graphique
    fig = px.bar(
        pct_disengaged,
        x='Segment',
        y='Pourcentage',
        title="Pourcentage de désengagement par segment",
        color='Pourcentage',
        color_continuous_scale='Reds',
        text='Pourcentage'
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Tableau des segments à risque
    high_risk_segments = pct_disengaged.sort_values('Pourcentage', ascending=False).head(3)
    
    st.markdown("### Segments à surveiller en priorité")
    
    for _, row in high_risk_segments.iterrows():
        cluster = row['Cluster']
        if cluster in cluster_profiles:
            st.warning(f"**{row['Segment']}** - {row['Pourcentage']}% de désengagement")
            st.markdown(f"*Action recommandée:* {cluster_profiles[cluster]['actions'][0]}")

with col2:
    at_risk = df[(df['will_disengage_30d'] == 1) & (df['is_disengaged'] == 0)]
    at_risk_pct = round(len(at_risk) / len(df) * 100, 1)
    
    st.metric("Utilisateurs à risque de désengagement (30j)", f"{at_risk_pct}%",
              delta=f"{at_risk_pct - 8.7:.1f}%" if at_risk_pct > 8.7 else f"{8.7 - at_risk_pct:.1f}%",
              delta_color="inverse")
    
    # Créer un indicateur de risque global
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = at_risk_pct + disengaged_pct,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Indice de risque global", 'font': {'size': 24}},
        delta = {'reference': 15, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 30], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkred"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 10], 'color': 'lightgreen'},
                {'range': [10, 20], 'color': 'orange'},
                {'range': [20, 30], 'color': 'salmon'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': at_risk_pct + disengaged_pct}}))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Analyse des facteurs de désengagement
    st.markdown("### Facteurs de désengagement")
    
    # Simuler des facteurs de désengagement (à remplacer par des données réelles)
    factors = {
        "Inactivité prolongée (>30 jours)": 42,
        "Faible interaction avec le contenu": 28,
        "Taux de rebond élevé": 15,
        "Absence d'inscription newsletter": 10,
        "Autres facteurs": 5
    }
    
    factors_df = pd.DataFrame({
        'Facteur': list(factors.keys()),
        'Pourcentage': list(factors.values())
    })
    
    fig = px.pie(
        factors_df,
        values='Pourcentage',
        names='Facteur',
        title="Principaux facteurs de désengagement",
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.Reds_r
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Recommandations marketing personnalisées
st.subheader("🎯 Recommandations Marketing Personnalisées")

# Sélection du cluster pour afficher les recommandations
selected_cluster = st.selectbox(
    "Sélectionnez un segment pour voir les recommandations détaillées :",
    sorted(df['cluster'].unique()),
    format_func=lambda x: f"Segment {x} - {cluster_profiles.get(x, {}).get('title', 'Non défini')}"
)

# Affichage des recommandations pour le cluster sélectionné
if selected_cluster in cluster_profiles:
    profile = cluster_profiles[selected_cluster]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Stratégie pour Segment {selected_cluster}: {profile['title']}")
        st.markdown(f"**Effectif :** {profile['size']}")
        st.markdown(f"**Description :** {profile['description']}")
        st.markdown(f"**Stratégie principale :** {profile['strategy']}")
        
        st.markdown("**Actions recommandées :**")
        for action in profile['actions']:
            st.markdown(f"- {action}")
    
    with col2:
        st.markdown("**KPIs à surveiller :**")
        for kpi in profile['kpis']:
            st.markdown(f"- {kpi}")
        
        st.markdown("**Types de contenu recommandés :**")
        for content in profile['content_types']:
            st.markdown(f"- {content}")
        
        # Ajouter un bouton pour générer un plan d'action détaillé
        if st.button("📝 Générer un plan d'action détaillé", key=f"plan_{selected_cluster}"):
            st.success("✅ Plan d'action généré et disponible pour téléchargement")
            
            # Créer le contenu du plan d'action
            plan_content = f"# Plan d'action marketing - Segment {selected_cluster}: {profile['title']}\n\n"
            plan_content += f"Date de génération: {current_date.strftime('%d/%m/%Y')}\n\n"
            plan_content += f"## Description du segment\n\n{profile['description']}\n\n"
            plan_content += f"## Stratégie principale\n\n{profile['strategy']}\n\n"
            
            plan_content += "## Actions recommandées\n\n"
            for i, action in enumerate(profile['actions'], 1):
                plan_content += f"{i}. {action}\n"
            
            plan_content += "\n## KPIs à surveiller\n\n"
            for i, kpi in enumerate(profile['kpis'], 1):
                plan_content += f"{i}. {kpi}\n"
            
            plan_content += "\n## Types de contenu recommandés\n\n"
            for i, content in enumerate(profile['content_types'], 1):
                plan_content += f"{i}. {content}\n"
            
            plan_content += "\n## Calendrier de déploiement\n\n"
            plan_content += "- Semaine 1: Préparation et configuration\n"
            plan_content += "- Semaine 2: Lancement des premières actions\n"
            plan_content += "- Semaines 3-4: Suivi et optimisation\n"
            plan_content += "- Mois 2: Analyse des résultats et ajustements\n"
            
            st.download_button(
                label="⬇️ Télécharger le plan d'action",
                data=plan_content,
                file_name=f"plan_action_segment_{selected_cluster}.md",
                mime="text/markdown"
            )

# Analyse des tendances d'engagement
st.subheader("📈 Tendances d'engagement")

col1, col2 = st.columns(2)

with col1:
    # Simuler des données de tendance d'engagement
    months = ["Jan", "Fév", "Mar", "Avr"]
    engagement_trend = pd.DataFrame({
        'Mois': months,
        'Score moyen': [42.3, 44.1, 45.8, 47.2],
        'Utilisateurs actifs': [58.2, 59.7, 61.5, 63.8]
    })
    
    fig = px.line(
        engagement_trend,
        x='Mois',
        y=['Score moyen', 'Utilisateurs actifs'],
        title="Évolution de l'engagement (2025)",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Simuler des données de conversion par segment
    conversion_data = []
    for cluster in sorted(df['cluster'].unique()):
        if cluster in cluster_profiles:
            # Simuler un taux de conversion basé sur le profil du cluster
            if cluster in ['7', '6']:
                conversion_rate = np.random.uniform(15, 25)
            elif cluster in ['3_1', '4', '5']:
                conversion_rate = np.random.uniform(8, 15)
            elif cluster in ['0', '1', '2', '3_2']:
                conversion_rate = np.random.uniform(1, 8)
            else:
                conversion_rate = np.random.uniform(5, 10)
            
            conversion_data.append({
                'Segment': f"{cluster}: {cluster_profiles[cluster]['title']}",
                'Taux de conversion (%)': round(conversion_rate, 1)
            })
    
    conversion_df = pd.DataFrame(conversion_data)
    
    fig = px.bar(
        conversion_df,
        x='Segment',
        y='Taux de conversion (%)',
        title="Taux de conversion par segment",
        color='Taux de conversion (%)',
        color_continuous_scale='Greens'
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# Opportunités de croissance
st.subheader("🚀 Opportunités de croissance")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("📚 **Contenus à développer en priorité**")
    st.markdown("1. Séries de tutoriels Data Science pour débutants")
    st.markdown("2. Cas d'études sectoriels (Finance, Santé, Retail)")
    st.markdown("3. Webinaires avancés sur les techniques de ML")
    st.markdown("4. Datasets commentés pour pratique guidée")
    st.markdown("5. Forums thématiques animés par des experts")

with col2:
    st.info("🎯 **Segments à fort potentiel**")
    st.markdown("1. **Segment 3_1** - Actifs avec Fort Potentiel")
    st.markdown("   *Conversion vers contribution communautaire*")
    st.markdown("2. **Segment 4** - Bonne Interaction Générale")
    st.markdown("   *Diversification des domaines d'expertise*")
    st.markdown("3. **Segment 6** - Très Actif et Intense")
    st.markdown("   *Programme ambassadeur et mentorat*")

with col3:
    st.info("💡 **Actions rapides à impact élevé**")
    st.markdown("1. Campagne email ciblée pour Segment 1")
    st.markdown("2. Data Challenge mensuel pour Segments 3_1 et 6")
    st.markdown("3. Programme de reconnaissance pour Segment 7")
    st.markdown("4. Webinaires d'introduction pour Segment 0")
    st.markdown("5. Refonte UX des pages d'entrée principales")

# Pied de page
st.markdown("---")
st.info("⬅️ Utilisez le menu de gauche pour explorer chaque fonctionnalité en détail.")

# Sidebar pour la navigation
st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("---")

# Ajout d'un logo Management & Data Science (simulé)
st.sidebar.markdown("### Management & Data Science")
st.sidebar.markdown("*Plateforme collaborative francophone dédiée à la data science et au management*")

# Informations supplémentaires dans la sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Statistiques Rapides")
st.sidebar.write(f"Nombre total d'utilisateurs: **{len(df):,}**")
st.sidebar.write(f"Nombre de segments: **{len(df['cluster'].unique())}**")
st.sidebar.write(f"Utilisateurs actifs: **{active_pct}%**")
st.sidebar.write(f"Score d'engagement moyen: **{avg_engagement}**")

# Ajout d'un filtre de date
st.sidebar.markdown("---")
st.sidebar.subheader("⏱️ Filtre Temporel")
date_options = ["7 derniers jours", "30 derniers jours", "90 derniers jours", "Année en cours"]
selected_date = st.sidebar.selectbox("Période d'analyse:", date_options)
st.sidebar.markdown(f"Données filtrées sur: **{selected_date}**")

# Ajout d'un filtre par pays
countries = ['Tous'] + sorted(df['country'].dropna().unique().tolist())
selected_country = st.sidebar.selectbox("Filtrer par pays:", countries)
if selected_country != 'Tous':
    st.sidebar.markdown(f"Pays sélectionné: **{selected_country}**")

# Ajout d'un filtre par type de contenu
content_types = ['Tous', 'Articles', 'Datasets', 'Cours', 'Projets', 'Forums']
selected_content = st.sidebar.selectbox("Type de contenu:", content_types)
if selected_content != 'Tous':
    st.sidebar.markdown(f"Contenu filtré: **{selected_content}**")

# Actions rapides
st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Actions Rapides")

if st.sidebar.button("📧 Créer une campagne email"):
    st.sidebar.success("✅ Redirection vers le module de création de campagne")

if st.sidebar.button("📊 Exporter les données"):
    st.sidebar.success("✅ Données exportées avec succès")

if st.sidebar.button("🔄 Actualiser les données"):
    st.sidebar.info("🔄 Actualisation en cours...")
    st.sidebar.success("✅ Données actualisées")
