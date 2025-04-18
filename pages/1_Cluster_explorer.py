import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Configuration de la page
st.set_page_config(
    page_title="🖱️ Cluster Explorer",
    page_icon="🎯",
    layout="wide"
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

# Chargement des données
df = load_data()
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

# Dictionnaire des profils de clusters détaillés
cluster_profiles = {
    '0': {
        'title': "Faiblement Engagé, Passif",
        'size': "3 849 utilisateurs",
        'description': "Utilisateurs avec un faible engagement (score moyen: 3,18) et peu d'interactions (1,86 clics/session). Principalement issus de recherche organique (62,8%).",
        'characteristics': [
            "Sessions moyennes : 1,44 avec faible interactivité",
            "Engagement global modéré à faible",
            "Trafic principalement issu de recherche organique",
            "Taux de rebond élevé"
        ],
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
        ]
    },
    '1': {
        'title': "Passif à Risque élevé de Désengagement",
        'size': "4 369 utilisateurs",
        'description': "Utilisateurs avec un engagement très limité (score moyen: 2,18) et des interactions modérées sans profondeur.",
        'characteristics': [
            "Engagement très limité (engagement score moyen : 2,18)",
            "Clics par session modérés (2,39) mais sans réelle profondeur d'interaction",
            "Forte probabilité de désengagement",
            "Faible diversité des pages consultées"
        ],
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
        ]
    },
    '2': {
        'title': "Très Faible Engagement",
        'size': "273 utilisateurs",
        'description': "Petit groupe d'utilisateurs avec un engagement très faible (score: 1,68) et peu d'interactions.",
        'characteristics': [
            "Engagement très faible (score 1,68)",
            "Interactions limitées (clic/session à 1,72)",
            "Faible volume (273 utilisateurs)",
            "Rebond rapide après visite"
        ],
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
        ]
    },
    '3_1': {
        'title': "Actif avec Fort Potentiel",
        'size': "7 974 utilisateurs",
        'description': "Grand groupe d'utilisateurs avec un bon engagement (score: 4,27) et une bonne interactivité (3,37 clics/session).",
        'characteristics': [
            "Engagement assez élevé (4,27)",
            "Bonne interactivité (clics/session : 3,37)",
            "Groupe important (7 974 utilisateurs)",
            "Fort potentiel d'approfondissement"
        ],
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
        ]
    },
    '3_2': {
        'title': "Moyennement Actif mais Sous-exploité",
        'size': "674 utilisateurs",
        'description': "Groupe modéré avec un engagement moyen (score: 3,46) et des interactions moyennes (2,00 clics/session).",
        'characteristics': [
            "Engagement modéré (3,46)",
            "Interactions moyennes (clic/session : 2,00)",
            "Groupe de taille moyenne (674 utilisateurs)",
            "Potentiel inexploité"
        ],
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
        ]
    },
    '4': {
        'title': "Bonne Interaction Générale",
        'size': "675 utilisateurs",
        'description': "Groupe modéré avec un engagement élevé (score: 4,80) mais des interactions moyennes (1,94 clics/session).",
        'characteristics': [
            "Engagement élevé (4,80)",
            "Interaction moyenne (clic/session : 1,94)",
            "Groupe de taille moyenne (675 utilisateurs)",
            "Potentiel de diversification"
        ],
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
        ]
    },
    '5': {
        'title': "Petit Groupe Actif",
        'size': "440 utilisateurs",
        'description': "Petit groupe engagé avec un bon potentiel d'approfondissement (score: 4,07).",
        'characteristics': [
            "Engagement modéré à fort (score : 4,07)",
            "Potentiel élevé pour approfondir l'interaction",
            "Petit groupe mais engagé (440 utilisateurs)",
            "Intérêts spécifiques et ciblés"
        ],
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
        ]
    },
    '6': {
        'title': "Très Actif et Intense",
        'size': "1 186 utilisateurs",
        'description': "Groupe significatif avec une forte intensité d'engagement (score: 3,85) et des interactions très élevées (6,13 clics/session).",
        'characteristics': [
            "Forte intensité d'engagement (3,85)",
            "Interactions très élevées (6,13 clics/session)",
            "Groupe significatif (1 186 utilisateurs)",
            "Utilisateurs très actifs et engagés"
        ],
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
        ]
    },
    '7': {
        'title': "Petit Groupe Ultra-Engagé",
        'size': "223 utilisateurs",
        'description': "Petit groupe d'élite avec un engagement exceptionnel (score: 4,80) et une très forte intensité (4,55).",
        'characteristics': [
            "Engagement exceptionnel (4,80)",
            "Très forte intensité (4,55)",
            "Petit groupe d'élite (223 utilisateurs)",
            "Utilisateurs les plus précieux de la plateforme"
        ],
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
        ]
    }
}

# En-tête de la page
st.title("🖱️ Cluster Explorer")
st.markdown("### Analyse détaillée des segments utilisateurs de Management & Data Science")

# Sélection du cluster à explorer
selected_cluster = st.selectbox(
    "Sélectionnez un cluster pour une analyse détaillée :",
    sorted(df['cluster'].unique()),
    format_func=lambda x: f"Cluster {x}"
)

# Filtrer les données pour le cluster sélectionné
df_cluster = df[df['cluster'] == selected_cluster]

# Affichage des métriques clés
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Effectif", len(df_cluster))
with col2:
    active_pct = round(df_cluster[df_cluster['is_active'] == 1].shape[0] / len(df_cluster) * 100, 1)
    st.metric("Utilisateurs actifs", f"{active_pct}%")
with col3:
    disengaged_pct = round(df_cluster[df_cluster['is_disengaged'] == 1].shape[0] / len(df_cluster) * 100, 1)
    st.metric("Utilisateurs désengagés", f"{disengaged_pct}%")
with col4:
    avg_engagement = round(df_cluster['engagement_score'].mean(), 2)
    st.metric("Score d'engagement moyen", f"{avg_engagement}")

# Affichage du profil du cluster sélectionné
if selected_cluster in cluster_profiles:
    profile = cluster_profiles[selected_cluster]
    
    st.subheader(f"📊 Profil détaillé : {profile['title']} ({profile['size']})")
    
    st.markdown(f"**Description :** {profile['description']}")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Caractéristiques principales :**")
        for char in profile['characteristics']:
            st.markdown(f"- {char}")
            
        # Statistiques descriptives du cluster
        st.markdown("**Statistiques détaillées :**")
        engagement_metrics = ['engagement_score', 'global_engagement_score', 'engagement_intensity', 
                            'multi_interaction_score', 'session_efficiency', 'retention_like_score']
        cluster_stats = df_cluster[engagement_metrics].describe().round(2)
        st.dataframe(cluster_stats, use_container_width=True)
    
    with col2:
        # Comparaison avec la moyenne globale
        mean_cluster = df_cluster[engagement_metrics].mean()
        mean_global = df[engagement_metrics].mean()
        
        comparison = pd.DataFrame({
            f'Cluster {selected_cluster}': mean_cluster,
            'Moyenne Globale': mean_global
        })
        
        # Création d'un graphique à barres pour la comparaison
        fig = px.bar(
            comparison.reset_index(),
            x='index',
            y=[f'Cluster {selected_cluster}', 'Moyenne Globale'],
            barmode='group',
            title=f"Comparaison Cluster {selected_cluster} vs Moyenne Globale",
            labels={'index': 'Métrique', 'value': 'Valeur', 'variable': 'Groupe'},
            color_discrete_sequence=['#3366CC', '#DC3912']
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Analyse des comportements utilisateurs
st.subheader("🔍 Comportements utilisateurs")

col1, col2 = st.columns(2)

with col1:
    # Distribution des sessions
    fig = px.histogram(
        df_cluster,
        x='nb_sessions',
        nbins=20,
        title="Distribution du nombre de sessions",
        labels={'nb_sessions': 'Nombre de sessions', 'count': 'Nombre d\'utilisateurs'},
        color_discrete_sequence=['#3366CC']
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Distribution des clics par session
    fig = px.histogram(
        df_cluster,
        x='clicks_per_session',
        nbins=20,
        title="Distribution des clics par session",
        labels={'clicks_per_session': 'Clics par session', 'count': 'Nombre d\'utilisateurs'},
        color_discrete_sequence=['#3366CC']
    )
    st.plotly_chart(fig, use_container_width=True)

# Analyse des sources de trafic
st.subheader("🌐 Sources de trafic et comportement")

col1, col2 = st.columns(2)

with col1:
    # Distribution par source de trafic
    if 'medium' in df_cluster.columns:
        medium_counts = df_cluster['medium'].value_counts().head(10)
        fig = px.pie(
            values=medium_counts.values,
            names=medium_counts.index,
            title="Principales sources de trafic",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Données sur les sources de trafic non disponibles")

with col2:
    # Relation entre engagement et nombre de sessions
    fig = px.scatter(
        df_cluster,
        x='nb_sessions',
        y='engagement_score',
        title="Relation entre nombre de sessions et engagement",
        labels={'nb_sessions': 'Nombre de sessions', 'engagement_score': 'Score d\'engagement'},
        color='is_active',
        color_discrete_sequence=['#DC3912', '#3366CC'],
        size='clicks_per_session',
        hover_data=['recency_days', 'bounce_rate']
    )
    st.plotly_chart(fig, use_container_width=True)

# Analyse temporelle
st.subheader("📅 Analyse temporelle")

col1, col2 = st.columns(2)

with col1:
    # Distribution par jour de la semaine
    weekday_counts = df_cluster['dayofweek'].value_counts().sort_index()
    weekday_mapping = {
        'Mon': 'Lundi', 
        'Tue': 'Mardi', 
        'Wed': 'Mercredi', 
        'Thu': 'Jeudi', 
        'Fri': 'Vendredi', 
        'Sat': 'Samedi', 
        'Sun': 'Dimanche'
    }
    weekday_counts.index = [weekday_mapping.get(day, day) for day in weekday_counts.index]

    fig = px.bar(
        x=weekday_counts.index,
        y=weekday_counts.values,
        title="Distribution par jour de la semaine",
        labels={'x': 'Jour de la semaine', 'y': 'Nombre d\'utilisateurs'},
        color_discrete_sequence=['#3366CC']
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Distribution par heure de la journée si disponible
    if 'hour' in df_cluster.columns:
        hour_counts = df_cluster['hour'].value_counts().sort_index()
        fig = px.line(
            x=hour_counts.index,
            y=hour_counts.values,
            title="Distribution par heure de la journée",
            labels={'x': 'Heure', 'y': 'Nombre d\'utilisateurs'},
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Distribution par récence
        fig = px.histogram(
            df_cluster,
            x='recency_days',
            nbins=30,
            title="Distribution par récence (jours depuis dernière visite)",
            labels={'recency_days': 'Jours depuis dernière visite', 'count': 'Nombre d\'utilisateurs'},
            color_discrete_sequence=['#3366CC']
        )
        st.plotly_chart(fig, use_container_width=True)

# Stratégie marketing recommandée
if selected_cluster in cluster_profiles:
    st.subheader("🎯 Stratégie marketing recommandée")
    
    profile = cluster_profiles[selected_cluster]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"**Stratégie principale :** {profile['strategy']}")
        
        st.markdown("**Actions recommandées :**")
        for action in profile['actions']:
            st.markdown(f"- {action}")
    
    with col2:
        st.markdown("**KPIs à surveiller :**")
        for kpi in profile['kpis']:
            st.markdown(f"- {kpi}")
        
        # Bouton pour générer un plan d'action détaillé
        if st.button("📝 Générer un plan d'action détaillé", key=f"plan_{selected_cluster}"):
            st.success("✅ Plan d'action généré et disponible pour téléchargement")
            st.download_button(
                label="⬇️ Télécharger le plan d'action",
                data=f"Plan d'action détaillé pour le segment {selected_cluster} - {profile['title']}\n\nStratégie: {profile['strategy']}\n\nActions recommandées:\n- {profile['actions'][0]}\n- {profile['actions'][1]}\n- {profile['actions'][2]}\n- {profile['actions'][3]}\n- {profile['actions'][4]}\n\nKPIs à surveiller:\n- {profile['kpis'][0]}\n- {profile['kpis'][1]}\n- {profile['kpis'][2]}\n- {profile['kpis'][3]}\n\nCalendrier de déploiement:\n- Semaine 1: Préparation et configuration\n- Semaine 2: Lancement des premières actions\n- Semaine 3-4: Suivi et optimisation\n- Mois 2: Analyse des résultats et ajustements",
                file_name=f"plan_action_cluster_{selected_cluster}.txt",
                mime="text/plain"
            )

# Contenu recommandé pour ce segment
st.subheader("📚 Contenu recommandé pour ce segment")

col1, col2, col3 = st.columns(3)

content_recommendations = {
    '0': [
        "Introduction à la Data Science : concepts fondamentaux",
        "Guide étape par étape pour débuter en analyse de données",
        "Témoignages : parcours de débutant à expert en data science"
    ],
    '1': [
        "Cas d'étude : Impact de la data science dans le secteur financier",
        "Les nouveautés incontournables en data science (2025)",
        "Mini-formation : 15 minutes par jour pour progresser en data science"
    ],
    '2': [
        "Data Science simplifiée : comprendre les concepts clés en 5 minutes",
        "Success stories : comment la data science a transformé ces entreprises",
        "Accès premium temporaire : découvrez nos contenus exclusifs"
    ],
    '3_1': [
        "Data Challenge : Analyse prédictive sur données réelles",
        "Webinaire avancé : Techniques de machine learning pour professionnels",
        "Parcours thématique : Spécialisation en NLP pour data scientists"
    ],
    '3_2': [
        "Sélection de datasets pour pratiquer vos compétences",
        "Communauté : rejoignez les discussions sur l'IA générative",
        "Mini-challenge : optimisation d'algorithmes de clustering"
    ],
    '4': [
        "Découvrez de nouveaux domaines : Data Science pour la finance",
        "Atelier pratique : partager vos connaissances avec la communauté",
        "Collaborations : projets ouverts recherchant des contributeurs"
    ],
    '5': [
        "Contenu spécialisé : Techniques avancées de feature engineering",
        "Groupe d'experts : rejoignez la discussion sur l'éthique en IA",
        "Publication : opportunités de partager votre expertise"
    ],
    '6': [
        "Accès anticipé : nouvelles fonctionnalités de la plateforme",
        "Programme de mentorat : devenez animateur thématique",
        "Événement exclusif : rencontre avec des experts du domaine"
    ],
    '7': [
        "Programme Ambassadeur : co-création de contenus premium",
        "Interview : partagez votre expertise avec notre communauté",
        "Événement VIP : table ronde sur l'avenir de la data science"
    ]
}

if selected_cluster in content_recommendations:
    recommendations = content_recommendations[selected_cluster]
    
    with col1:
        st.info(f"📖 **Article recommandé**\n\n{recommendations[0]}")
        if st.button("Consulter l'article", key="article"):
            st.success("✅ Article ouvert dans un nouvel onglet!")
    
    with col2:
        st.info(f"🎓 **Formation recommandée**\n\n{recommendations[1]}")
        if st.button("Accéder à la formation", key="formation"):
            st.success("✅ Formation ouverte dans un nouvel onglet!")
    
    with col3:
        st.info(f"🤝 **Opportunité de participation**\n\n{recommendations[2]}")
        if st.button("Participer", key="participation"):
            st.success("✅ Inscription confirmée!")

# Actions marketing
st.subheader("🚀 Actions marketing à déployer")

col1, col2, col3 = st.columns(3)

email_templates = {
    '0': {
        'subject': "🔍 Découvrez les fondamentaux de la Data Science en 5 minutes",
        'content': "Bonjour,\n\nNous avons sélectionné pour vous une introduction aux concepts fondamentaux de la Data Science, adaptée à votre niveau.\n\nDécouvrez notre série de contenus 'Data Science pour tous' et commencez votre parcours d'apprentissage dès aujourd'hui."
    },
    '1': {
        'subject': "🔄 Voici ce que vous avez manqué sur Management & Data Science",
        'content': "Bonjour,\n\nDepuis votre dernière visite, notre communauté a partagé de nombreux contenus qui pourraient vous intéresser.\n\nVoici une sélection personnalisée basée sur vos centres d'intérêt :\n\n- Cas d'étude : Impact de la Data Science dans votre secteur\n- Nouveaux datasets disponibles pour vos analyses\n- Discussions actives sur les sujets qui vous passionnent"
    },
    '2': {
        'subject': "🎁 Accès premium offert : Redécouvrez Management & Data Science",
        'content': "Bonjour,\n\nNous vous offrons un accès temporaire à tous nos contenus premium pour vous permettre de redécouvrir la valeur de Management & Data Science.\n\nVoici ce que vous pourrez explorer :\n\n- Études de cas complètes avec code source\n- Datasets exclusifs pour vos analyses\n- Formations vidéo sur les techniques avancées"
    },
    '3_1': {
        'subject': "🏆 Rejoignez notre programme Data Leaders et partagez votre expertise",
        'content': "Bonjour,\n\nVotre activité sur Management & Data Science vous place parmi nos utilisateurs les plus engagés.\n\nNous vous invitons à rejoindre notre programme Data Leaders qui vous permettra de :\n\n- Participer à nos Data Challenges exclusifs\n- Obtenir des badges de reconnaissance pour vos contributions\n- Accéder en avant-première à nos nouveaux contenus"
    },
    '3_2': {
        'subject': "📊 Ressources recommandées pour approfondir vos connaissances",
        'content': "Bonjour,\n\nBasé sur votre activité récente, nous avons sélectionné des ressources qui pourraient vous aider à approfondir vos connaissances en data science.\n\nVoici nos recommandations personnalisées :\n\n- Articles intermédiaires sur les techniques de clustering\n- Datasets récents dans votre domaine\n- Mini-challenge adapté à votre niveau"
    },
    '4': {
        'subject': "🌟 Explorez de nouveaux domaines de la Data Science",
        'content': "Bonjour,\n\nVotre engagement sur Management & Data Science est excellent, et nous pensons que vous pourriez être intéressé par l'exploration de nouveaux domaines connexes.\n\nVoici des suggestions basées sur votre profil :\n\n- Data Science pour la Finance: Articles et ressources\n- NLP: Projets pratiques pour développer vos compétences\n- Opportunités de partager votre expertise dans vos domaines de prédilection"
    },
    '5': {
        'subject': "👑 Invitation à rejoindre notre groupe d'experts",
        'content': "Bonjour,\n\nVotre expertise et votre engagement sur Management & Data Science vous distinguent.\n\nNous vous invitons à rejoindre notre groupe d'experts où vous pourrez :\n\n- Participer à des discussions spécialisées avec d'autres experts\n- Contribuer à la création de contenus de niche\n- Être reconnu comme expert dans votre domaine"
    },
    '6': {
        'subject': "🔑 Accès VIP : Programme Data Science Influencers",
        'content': "Bonjour,\n\nFélicitations ! Votre activité exceptionnelle sur Management & Data Science vous qualifie pour notre programme VIP 'Data Science Influencers'.\n\nVos avantages exclusifs :\n\n- Accès anticipé aux nouvelles fonctionnalités et contenus\n- Invitations à des événements privés avec des experts du domaine\n- Opportunités de collaboration sur des projets stratégiques\n- Badge VIP visible sur votre profil"
    },
    '7': {
        'subject': "🌠 Devenez Ambassadeur Management & Data Science",
        'content': "Bonjour,\n\nVous faites partie du cercle très restreint de nos utilisateurs les plus engagés et nous souhaitons vous proposer une relation privilégiée avec notre équipe.\n\nEn tant qu'Ambassadeur, vous bénéficierez de :\n\n- Un canal de communication direct avec notre équipe de direction\n- Des opportunités de co-création de contenus et fonctionnalités\n- Une mise en avant de votre expertise via des interviews et témoignages\n- Des avantages exclusifs réservés aux Ambassadeurs"
    }
}

with col1:
    st.info("📧 **Email Marketing**\n\nEnvoyez des campagnes ciblées à ce segment")
    
    if selected_cluster in email_templates:
        template = email_templates[selected_cluster]
        st.markdown(f"**Objet suggéré :** {template['subject']}")
        
    if st.button("Créer une campagne Email"):
        st.success("✅ Modèle de campagne email créé pour ce segment!")
        if selected_cluster in email_templates:
            template = email_templates[selected_cluster]
            st.code(f"Objet: {template['subject']}\n\n{template['content']}", language="text")

with col2:
    st.info("📱 **SMS & Notifications**\n\nCommuniquez rapidement avec ce segment")
    
    sms_templates = {
        '0': "Management & Data Science: Découvrez notre nouvelle série 'Data Science pour tous'. Cliquez ici pour accéder à votre premier module gratuit.",
        '1': "Management & Data Science: 3 nouveaux cas d'études dans votre domaine viennent d'être publiés. Accédez-y en priorité.",
        '2': "Management & Data Science vous offre 7 jours d'accès premium ! Activez votre compte maintenant et découvrez nos contenus exclusifs.",
        '3_1': "Management & Data Science: Un nouveau Data Challenge vient d'être lancé dans votre domaine d'expertise. Participez avant le 25/04.",
        '3_2': "Management & Data Science: 5 nouveaux datasets dans votre domaine viennent d'être publiés. Explorez-les et partagez vos analyses.",
        '4': "Management & Data Science: Découvrez 3 nouveaux domaines connexes à vos intérêts. Élargissez vos compétences dès maintenant.",
        '5': "Management & Data Science: Vous êtes invité à rejoindre notre groupe d'experts. Accédez à des contenus exclusifs et partagez votre expertise.",
        '6': "Management & Data Science: Félicitations! Vous êtes invité à notre événement VIP le 28/04. Rencontrez les experts et influenceurs de la Data Science.",
        '7': "Management & Data Science: Vous êtes invité à rejoindre notre comité consultatif. Participez aux décisions stratégiques et à l'évolution de la plateforme."
    }
    
    if selected_cluster in sms_templates:
        st.markdown(f"**Message suggéré :**\n\n{sms_templates[selected_cluster]}")
    
    if st.button("Créer une campagne SMS"):
        st.success("✅ Modèle de campagne SMS créé pour ce segment!")
        if selected_cluster in sms_templates:
            st.code(sms_templates[selected_cluster], language="text")

with col3:
    st.info("🎁 **Offres spéciales**\n\nCréez des offres adaptées à ce segment")
    
    offer_templates = {
        '0': "Accès gratuit à notre parcours d'introduction 'Data Science pour tous' (valeur 49€)",
        '1': "50% de réduction sur notre pack de cas d'études sectoriels + accès à notre communauté",
        '2': "7 jours d'accès premium gratuit + 30% de réduction sur l'abonnement annuel",
        '3_1': "Accès prioritaire à nos Data Challenges + badge 'Data Leader' sur votre profil",
        '3_2': "Pack de 10 datasets premium + accès aux discussions privées entre membres",
        '4': "Accès à 3 domaines spécialisés de votre choix + 1 session de mentorat offerte",
        '5': "Statut 'Expert reconnu' + opportunité de publication sur notre blog",
        '6': "Programme VIP 'Data Science Influencers' + invitation à notre événement annuel",
        '7': "Programme 'Ambassadeur' exclusif + co-création de contenu rémunérée"
    }
    
    if selected_cluster in offer_templates:
        st.markdown(f"**Offre suggérée :**\n\n{offer_templates[selected_cluster]}")
    
    if st.button("Générer une offre spéciale"):
        st.success("✅ Offre spéciale générée pour ce segment!")
        if selected_cluster in offer_templates:
            st.code(offer_templates[selected_cluster], language="text")
