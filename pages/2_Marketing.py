import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib

# Configuration de la page
st.set_page_config(
    page_title="🚀 Marketing Automation",
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

# Dictionnaire des recommandations par cluster
cluster_recommendations = {
    '0': {
        'title': "Faiblement Engagé, Passif",
        'strategy': "Activation et éducation progressive",
        'description': "Utilisateurs avec un faible engagement (score moyen: 3,18) et peu d'interactions (1,86 clics/session). Principalement issus de recherche organique.",
        'email_subject': "🔍 Découvrez les fondamentaux de la Data Science en 5 minutes",
        'email_content': "Bonjour,\n\nNous avons sélectionné pour vous une introduction aux concepts fondamentaux de la Data Science, adaptée à votre niveau.\n\nDécouvrez notre série de contenus 'Data Science pour tous' et commencez votre parcours d'apprentissage dès aujourd'hui :\n[LIEN_PARCOURS]\n\nCordialement,\nL'équipe Management & Data Science",
        'sms_content': "Management & Data Science: Découvrez notre nouvelle série 'Data Science pour tous'. Cliquez ici pour accéder à votre premier module gratuit: [LIEN]",
        'notification_title': "Votre parcours Data Science vous attend !",
        'notification_message': "Commencez avec les fondamentaux dès maintenant"
    },
    '1': {
        'title': "Passif à Risque élevé de Désengagement",
        'strategy': "Réactivation ciblée et démonstration de valeur",
        'description': "Utilisateurs avec un engagement très limité (score moyen: 2,18) et des interactions modérées sans profondeur.",
        'email_subject': "🔄 Voici ce que vous avez manqué sur Management & Data Science",
        'email_content': "Bonjour,\n\nDepuis votre dernière visite, notre communauté a partagé de nombreux contenus qui pourraient vous intéresser.\n\nVoici une sélection personnalisée basée sur vos centres d'intérêt :\n\n- Cas d'étude : Impact de la Data Science dans votre secteur\n- Nouveaux datasets disponibles pour vos analyses\n- Discussions actives sur les sujets qui vous passionnent\n\nRedécouvrez Management & Data Science :\n[LIEN_SELECTION]\n\nCordialement,\nL'équipe Management & Data Science",
        'sms_content': "Management & Data Science: 3 nouveaux cas d'études dans votre domaine viennent d'être publiés. Accédez-y en priorité: [LIEN]",
        'notification_title': "Nouveaux contenus dans vos domaines !",
        'notification_message': "3 ressources pertinentes vous attendent"
    },
    '2': {
        'title': "Très Faible Engagement",
        'strategy': "Reconquête intensive avec proposition de valeur claire",
        'description': "Petit groupe d'utilisateurs (273) avec un engagement très faible (score: 1,68) et peu d'interactions.",
        'email_subject': "🎁 Accès premium offert : Redécouvrez Management & Data Science",
        'email_content': "Bonjour,\n\nNous vous offrons un accès temporaire à tous nos contenus premium pour vous permettre de redécouvrir la valeur de Management & Data Science.\n\nVoici ce que vous pourrez explorer :\n\n- Études de cas complètes avec code source\n- Datasets exclusifs pour vos analyses\n- Formations vidéo sur les techniques avancées\n\nVotre accès premium est valable 7 jours :\n[ACTIVER_MON_ACCES]\n\nCordialement,\nL'équipe Management & Data Science",
        'sms_content': "Management & Data Science vous offre 7 jours d'accès premium ! Activez votre compte maintenant et découvrez nos contenus exclusifs: [LIEN]",
        'notification_title': "7 jours d'accès premium offerts !",
        'notification_message': "Activez votre offre exclusive maintenant"
    },
    '3_1': {
        'title': "Actif avec Fort Potentiel",
        'strategy': "Approfondissement et engagement communautaire",
        'description': "Grand groupe d'utilisateurs (7 974) avec un bon engagement (score: 4,27) et une bonne interactivité (3,37 clics/session).",
        'email_subject': "🏆 Rejoignez notre programme Data Leaders et partagez votre expertise",
        'email_content': "Bonjour,\n\nVotre activité sur Management & Data Science vous place parmi nos utilisateurs les plus engagés.\n\nNous vous invitons à rejoindre notre programme Data Leaders qui vous permettra de :\n\n- Participer à nos Data Challenges exclusifs\n- Obtenir des badges de reconnaissance pour vos contributions\n- Accéder en avant-première à nos nouveaux contenus\n\nVotre parcours thématique personnalisé vous attend :\n[ACCEDER_AU_PROGRAMME]\n\nCordialement,\nL'équipe Management & Data Science",
        'sms_content': "Management & Data Science: Un nouveau Data Challenge vient d'être lancé dans votre domaine d'expertise. Participez avant le 25/04: [LIEN]",
        'notification_title': "Nouveau Data Challenge disponible !",
        'notification_message': "Participez et montrez votre expertise"
    },
    '3_2': {
        'title': "Moyennement Actif mais Sous-exploité",
        'strategy': "Activation ciblée et personnalisation accrue",
        'description': "Groupe modéré (674 utilisateurs) avec un engagement moyen (score: 3,46) et des interactions moyennes (2,00 clics/session).",
        'email_subject': "📊 Ressources recommandées pour approfondir vos connaissances",
        'email_content': "Bonjour,\n\nBasé sur votre activité récente, nous avons sélectionné des ressources qui pourraient vous aider à approfondir vos connaissances en data science.\n\nVoici nos recommandations personnalisées :\n\n- Articles intermédiaires sur [SUJET_INTERET]\n- Datasets récents dans votre domaine\n- Mini-challenge adapté à votre niveau\n\nDécouvrez ces ressources :\n[LIEN_RESSOURCES]\n\nCordialement,\nL'équipe Management & Data Science",
        'sms_content': "Management & Data Science: 5 nouveaux datasets dans votre domaine viennent d'être publiés. Explorez-les et partagez vos analyses: [LIEN]",
        'notification_title': "Nouvelles ressources pour vous !",
        'notification_message': "Contenus intermédiaires dans vos domaines"
    },
    '4': {
        'title': "Bonne Interaction Générale",
        'strategy': "Diversification des interactions et approfondissement",
        'description': "Groupe modéré (675 utilisateurs) avec un engagement élevé (score: 4,80) mais des interactions moyennes (1,94 clics/session).",
        'email_subject': "🌟 Explorez de nouveaux domaines de la Data Science",
        'email_content': "Bonjour,\n\nVotre engagement sur Management & Data Science est excellent, et nous pensons que vous pourriez être intéressé par l'exploration de nouveaux domaines connexes.\n\nVoici des suggestions basées sur votre profil :\n\n- [NOUVEAU_DOMAINE_1]: Articles et ressources pour débutants\n- [NOUVEAU_DOMAINE_2]: Projets pratiques pour développer vos compétences\n- Opportunités de partager votre expertise dans vos domaines de prédilection\n\nCommencez votre exploration :\n[LIEN_EXPLORER]\n\nCordialement,\nL'équipe Management & Data Science",
        'sms_content': "Management & Data Science: Découvrez 3 nouveaux domaines connexes à vos intérêts. Élargissez vos compétences dès maintenant: [LIEN]",
        'notification_title': "Élargissez vos horizons !",
        'notification_message': "Nouveaux domaines à explorer"
    },
    '5': {
        'title': "Petit Groupe Actif",
        'strategy': "Spécialisation et valorisation des contributions",
        'description': "Petit groupe engagé (440 utilisateurs) avec un bon potentiel d'approfondissement (score: 4,07).",
        'email_subject': "👑 Invitation à rejoindre notre groupe d'experts",
        'email_content': "Bonjour,\n\nVotre expertise et votre engagement sur Management & Data Science vous distinguent.\n\nNous vous invitons à rejoindre notre groupe d'experts où vous pourrez :\n\n- Participer à des discussions spécialisées avec d'autres experts\n- Contribuer à la création de contenus de niche\n- Être reconnu comme expert dans votre domaine\n\nRejoignez le groupe dès maintenant :\n[LIEN_GROUPE_EXPERTS]\n\nCordialement,\nL'équipe Management & Data Science",
        'sms_content': "Management & Data Science: Vous êtes invité à rejoindre notre groupe d'experts. Accédez à des contenus exclusifs et partagez votre expertise: [LIEN]",
        'notification_title': "Invitation groupe d'experts !",
        'notification_message': "Rejoignez l'élite de notre communauté"
    },
    '6': {
        'title': "Très Actif et Intense",
        'strategy': "Valorisation premium et leadership communautaire",
        'description': "Groupe significatif (1 186 utilisateurs) avec une forte intensité d'engagement (score: 3,85) et des interactions très élevées (6,13 clics/session).",
        'email_subject': "🔑 Accès VIP : Programme Data Science Influencers",
        'email_content': "Bonjour,\n\nFélicitations ! Votre activité exceptionnelle sur Management & Data Science vous qualifie pour notre programme VIP 'Data Science Influencers'.\n\nVos avantages exclusifs :\n\n- Accès anticipé aux nouvelles fonctionnalités et contenus\n- Invitations à des événements privés avec des experts du domaine\n- Opportunités de collaboration sur des projets stratégiques\n- Badge VIP visible sur votre profil\n\nActivez votre statut VIP :\n[ACTIVER_STATUT_VIP]\n\nCordialement,\nL'équipe Management & Data Science",
        'sms_content': "Management & Data Science: Félicitations! Vous êtes invité à notre événement VIP le 28/04. Rencontrez les experts et influenceurs de la Data Science: [LIEN]",
        'notification_title': "Votre statut VIP vous attend !",
        'notification_message': "Activez vos avantages exclusifs"
    },
    '7': {
        'title': "Petit Groupe Ultra-Engagé",
        'strategy': "Partenariat stratégique et co-création",
        'description': "Petit groupe d'élite (223 utilisateurs) avec un engagement exceptionnel (score: 4,80) et une très forte intensité (4,55).",
        'email_subject': "🌠 Devenez Ambassadeur Management & Data Science",
        'email_content': "Bonjour,\n\nVous faites partie du cercle très restreint de nos utilisateurs les plus engagés et nous souhaitons vous proposer une relation privilégiée avec notre équipe.\n\nEn tant qu'Ambassadeur, vous bénéficierez de :\n\n- Un canal de communication direct avec notre équipe de direction\n- Des opportunités de co-création de contenus et fonctionnalités\n- Une mise en avant de votre expertise via des interviews et témoignages\n- Des avantages exclusifs réservés aux Ambassadeurs\n\nAcceptez votre nomination :\n[DEVENIR_AMBASSADEUR]\n\nCordialement,\nL'équipe Management & Data Science",
        'sms_content': "Management & Data Science: Vous êtes invité à rejoindre notre comité consultatif. Participez aux décisions stratégiques et à l'évolution de la plateforme: [LIEN]",
        'notification_title': "Invitation exclusive !",
        'notification_message': "Devenez Ambassadeur de notre plateforme"
    }
}

# En-tête de la page
st.title("🚀 Marketing Automation")
st.markdown("### Automatisation des actions marketing par segment")

# Sélection du type de campagne
campaign_type = st.radio(
    "Sélectionnez le type de campagne :",
    ["Email", "SMS", "Push Notification"],
    horizontal=True
)

# Sélection du segment cible
st.subheader("🎯 Ciblage")

col1, col2 = st.columns(2)

with col1:
    target_option = st.radio(
        "Cibler par :",
        ["Cluster", "Niveau d'engagement", "Risque de désengagement", "Pays"]
    )
    
    if target_option == "Cluster":
        clusters = sorted(df['cluster'].unique())
        target_value = st.multiselect(
            "Sélectionnez les clusters :",
            clusters,
            default=[clusters[0]] if len(clusters) > 0 else []
        )
        filtered_df = df[df['cluster'].isin(target_value)]
        
        # Afficher les informations sur le cluster sélectionné
        if len(target_value) == 1 and target_value[0] in cluster_recommendations:
            cluster_info = cluster_recommendations[target_value[0]]
            st.info(f"**{cluster_info['title']}**: {cluster_info['description']}")
            st.markdown(f"**Stratégie recommandée**: {cluster_info['strategy']}")
    
    elif target_option == "Niveau d'engagement":
        engagement_min = st.slider(
            "Score d'engagement minimum :",
            min_value=0,
            max_value=100,
            value=50
        )
        filtered_df = df[df['engagement_score'] >= engagement_min]
    
    elif target_option == "Risque de désengagement":
        target_value = st.radio(
            "Cibler les utilisateurs :",
            ["À risque de désengagement", "Déjà désengagés", "Actifs et fidèles"]
        )
        
        if target_value == "À risque de désengagement":
            filtered_df = df[(df['will_disengage_30d'] == 1) & (df['is_disengaged'] == 0)]
        elif target_value == "Déjà désengagés":
            filtered_df = df[df['is_disengaged'] == 1]
        else:
            filtered_df = df[(df['is_active'] == 1) & (df['is_disengaged'] == 0)]
    
    else:  # Pays
        countries = sorted(df['country'].dropna().unique().tolist())
        target_value = st.multiselect(
            "Sélectionnez les pays :",
            countries,
            default=countries[:1] if countries else []
        )
        filtered_df = df[df['country'].isin(target_value)]

with col2:
    # Affichage des statistiques de ciblage
    st.metric("Nombre d'utilisateurs ciblés", len(filtered_df))
    
    # Répartition par cluster des utilisateurs ciblés
    cluster_counts = filtered_df['cluster'].value_counts().sort_index()
    
    fig = px.pie(
        values=cluster_counts.values,
        names=cluster_counts.index,
        title="Répartition par cluster des utilisateurs ciblés",
        labels={'names': 'Cluster', 'values': 'Nombre d\'utilisateurs'}
    )
    st.plotly_chart(fig, use_container_width=True)

# Configuration de la campagne
st.subheader("✏️ Contenu de la campagne")

# Déterminer si on peut proposer un contenu personnalisé
can_suggest_content = (
    target_option == "Cluster" and 
    len(target_value) == 1 and 
    target_value[0] in cluster_recommendations
)

if campaign_type == "Email":
    col1, col2 = st.columns(2)
    
    with col1:
        # Suggérer un objet d'email basé sur le cluster si possible
        default_subject = cluster_recommendations[target_value[0]]['email_subject'] if can_suggest_content else "🎉 Offre exclusive pour vous !"
        email_subject = st.text_input("Objet de l'email", default_subject)
        
        email_sender = st.text_input("Nom de l'expéditeur", "Management & Data Science")
        email_template = st.selectbox(
            "Template d'email",
            ["Template standard", "Newsletter", "Promotion", "Notification", "Bienvenue"]
        )
    
    with col2:
        # Suggérer un contenu d'email basé sur le cluster si possible
        default_content = cluster_recommendations[target_value[0]]['email_content'] if can_suggest_content else "Bonjour,\n\nNous sommes ravis de vous proposer une offre exclusive adaptée à vos préférences.\n\nCliquez sur le lien ci-dessous pour en profiter :\n[LIEN_OFFRE]\n\nCordialement,\nL'équipe Management & Data Science"
        email_content = st.text_area(
            "Contenu de l'email",
            default_content
        )
        
        st.color_picker("Couleur principale", "#3366CC")  # Couleur plus adaptée à une plateforme data science
        
        include_unsubscribe = st.checkbox("Inclure un lien de désabonnement", value=True)

elif campaign_type == "SMS":
    # Suggérer un contenu SMS basé sur le cluster si possible
    default_sms = cluster_recommendations[target_value[0]]['sms_content'] if can_suggest_content else "Management & Data Science: Découvrez nos nouveaux contenus et ressources adaptés à votre profil. Cliquez ici pour y accéder: [LIEN]"
    sms_content = st.text_area(
        "Contenu du SMS (160 caractères max)",
        default_sms
    )
    
    # Compteur de caractères
    st.write(f"Nombre de caractères : {len(sms_content)}/160")
    
    if len(sms_content) > 160:
        st.warning("⚠️ Le SMS dépasse la limite de 160 caractères.")

else:  # Push Notification
    col1, col2 = st.columns(2)
    
    with col1:
        # Suggérer un titre de notification basé sur le cluster si possible
        default_title = cluster_recommendations[target_value[0]]['notification_title'] if can_suggest_content else "Nouvelle ressource disponible !"
        notif_title = st.text_input("Titre de la notification", default_title)
        
        notif_icon = st.selectbox(
            "Icône",
            ["📊", "📈", "🔍", "📚", "💡", "🧠"]  # Icônes plus adaptées à une plateforme data science
        )
    
    with col2:
        # Suggérer un message de notification basé sur le cluster si possible
        default_message = cluster_recommendations[target_value[0]]['notification_message'] if can_suggest_content else "Accédez à du contenu personnalisé pour vous"
        notif_message = st.text_area(
            "Message (50 caractères max)",
            default_message
        )
        
        # Compteur de caractères
        st.write(f"Nombre de caractères : {len(notif_message)}/50")
        
        if len(notif_message) > 50:
            st.warning("⚠️ La notification dépasse la limite de 50 caractères.")
    
    deep_link = st.text_input("Deep link (URL de destination)", "https://management-data-science.com/ressources-personnalisees")

# Planification
st.subheader("📅 Planification")

col1, col2 = st.columns(2)

with col1:
    schedule_option = st.radio(
        "Quand envoyer la campagne ?",
        ["Immédiatement", "Planifier pour plus tard", "Envoi récurrent"]
    )
    
    if schedule_option == "Planifier pour plus tard":
        scheduled_date = st.date_input("Date d'envoi")
        scheduled_time = st.time_input("Heure d'envoi")
    
    elif schedule_option == "Envoi récurrent":
        recurrence = st.selectbox(
            "Fréquence",
            ["Quotidien", "Hebdomadaire", "Mensuel"]
        )
        
        if recurrence == "Hebdomadaire":
            weekdays = st.multiselect(
                "Jours de la semaine",
                ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"],
                default=["Lundi"]
            )
        
        elif recurrence == "Mensuel":
            monthdays = st.multiselect(
                "Jours du mois",
                list(range(1, 32)),
                default=[1]
            )

with col2:
    # Options avancées
    st.subheader("Options avancées")
    
    test_campaign = st.checkbox("Envoyer un test avant la campagne finale", value=True)
    
    if test_campaign:
        test_email = st.text_input("Email de test", "votre.email@exemple.com")
    
    analytics = st.checkbox("Activer le suivi des performances", value=True)
    
    if analytics:
        tracking_options = st.multiselect(
            "Options de suivi",
            ["Taux d'ouverture", "Clics", "Conversions", "Désabonnements", "Temps de lecture", "Partages"],
            default=["Taux d'ouverture", "Clics"]
        )

# Recommandations marketing spécifiques
if target_option == "Cluster" and len(target_value) == 1 and target_value[0] in cluster_recommendations:
    st.subheader("🎯 Recommandations marketing spécifiques")
    
    cluster_info = cluster_recommendations[target_value[0]]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Segment cible :** {cluster_info['title']}")
        st.markdown(f"**Stratégie :** {cluster_info['strategy']}")
        
        st.markdown("**Actions recommandées :**")
        if target_value[0] == '0':
            st.markdown("- Créer une série de contenus d'introduction 'Data Science pour tous'")
            st.markdown("- Proposer un parcours guidé de découverte des ressources fondamentales")
            st.markdown("- Mettre en avant des témoignages d'utilisateurs ayant progressé")
        elif target_value[0] == '1':
            st.markdown("- Proposer des cas d'études concrets montrant l'impact dans leur secteur")
            st.markdown("- Mettre en avant les nouveaux contenus depuis leur dernière visite")
            st.markdown("- Créer des parcours de micro-apprentissage à faible engagement initial")
        elif target_value[0] == '2':
            st.markdown("- Offrir un accès temporaire à des contenus premium pour démontrer la valeur")
            st.markdown("- Proposer des formats de contenu ultra-simplifiés et accessibles")
            st.markdown("- Mettre en avant les success stories de la communauté pour inspirer l'engagement")
        elif target_value[0] == '3_1':
            st.markdown("- Proposer des parcours thématiques avancés basés sur leurs centres d'intérêt")
            st.markdown("- Encourager la participation aux Data Challenges et projets collaboratifs")
            st.markdown("- Inviter à contribuer à la communauté par des partages d'expérience")
        elif target_value[0] == '3_2':
            st.markdown("- Proposer des contenus intermédiaires adaptés à leur niveau de compétence")
            st.markdown("- Mettre en avant les discussions et projets actifs dans leurs domaines d'intérêt")
            st.markdown("- Suggérer des connexions avec d'autres membres aux intérêts similaires")
        elif target_value[0] == '4':
            st.markdown("- Proposer l'exploration de nouveaux domaines connexes à leurs intérêts")
            st.markdown("- Encourager le partage de leurs connaissances via des contributions")
            st.markdown("- Suggérer des collaborations avec d'autres membres actifs")
        elif target_value[0] == '5':
            st.markdown("- Proposer des contenus de niche et avancés dans leurs domaines de prédilection")
            st.markdown("- Encourager la création et le partage de contenus spécialisés")
            st.markdown("- Mettre en avant leur expertise auprès de la communauté")
        elif target_value[0] == '6':
            st.markdown("- Offrir un accès anticipé aux nouvelles fonctionnalités et contenus")
            st.markdown("- Proposer des rôles de mentors ou d'animateurs thématiques")
            st.markdown("- Créer des expériences exclusives et personnalisées")
        elif target_value[0] == '7':
            st.markdown("- Proposer une relation privilégiée avec l'équipe de Management & Data Science")
            st.markdown("- Offrir des opportunités de co-création de contenus et fonctionnalités")
            st.markdown("- Valoriser leur expertise à travers des interviews et témoignages")
    
    with col2:
        st.markdown("**KPIs à surveiller :**")
        if target_value[0] == '0':
            st.markdown("- Taux de rebond sur les pages d'introduction")
            st.markdown("- Taux de progression dans le parcours guidé")
            st.markdown("- Taux de conversion vers l'inscription complète")
        elif target_value[0] == '1':
            st.markdown("- Taux de réactivation après campagne")
            st.markdown("- Taux d'ouverture des emails et clics")
            st.markdown("- Augmentation du temps passé sur la plateforme")
        elif target_value[0] == '2':
            st.markdown("- Taux d'activation de l'offre premium")
            st.markdown("- Taux de rétention après la période d'essai")
            st.markdown("- Augmentation du nombre de sessions")
        elif target_value[0] == '3_1':
            st.markdown("- Taux de participation aux Data Challenges")
            st.markdown("- Nombre de contributions à la communauté")
            st.markdown("- Taux d'engagement avec le contenu avancé")
        elif target_value[0] == '3_2':
            st.markdown("- Taux de clics sur les contenus recommandés")
            st.markdown("- Augmentation du nombre d'interactions par session")
            st.markdown("- Taux de participation aux mini-challenges")
        elif target_value[0] == '4':
            st.markdown("- Diversité des contenus consultés")
            st.markdown("- Taux de contribution (commentaires, évaluations)")
            st.markdown("- Nombre de nouvelles connexions établies")
        elif target_value[0] == '5':
            st.markdown("- Taux de participation aux groupes spécialisés")
            st.markdown("- Nombre et qualité des contenus créés")
            st.markdown("- Influence sur la communauté (partages, mentions)")
        elif target_value[0] == '6':
            st.markdown("- Taux d'activation du statut VIP")
            st.markdown("- Participation aux événements exclusifs")
            st.markdown("- Impact des contributions (vues, partages)")
        elif target_value[0] == '7':
            st.markdown("- Taux d'acceptation du programme Ambassadeur")
            st.markdown("- Qualité et impact des co-créations")
            st.markdown("- Influence sur l'acquisition de nouveaux membres")
        
        # Bouton pour générer un plan d'action détaillé
        if st.button("📝 Générer un plan d'action détaillé", key=f"plan_{target_value[0]}"):
            st.success("✅ Plan d'action généré et disponible pour téléchargement")
            st.download_button(
                label="⬇️ Télécharger le plan d'action",
                data=f"Plan d'action détaillé pour le segment {target_value[0]} - {cluster_info['title']}\n\nStratégie: {cluster_info['strategy']}\n\nActions recommandées:\n- Action 1\n- Action 2\n- Action 3\n\nCalendrier de déploiement:\n- Semaine 1: Préparation\n- Semaine 2: Lancement\n- Semaine 3-4: Suivi et optimisation",
                file_name=f"plan_action_cluster_{target_value[0]}.txt",
                mime="text/plain"
            )

# Bouton d'envoi
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🚀 Lancer la campagne", use_container_width=True):
        st.balloons()
        st.success(f"✅ Campagne {campaign_type} créée avec succès pour {len(filtered_df)} utilisateurs !")
        
        # Affichage d'un récapitulatif
        st.subheader("Récapitulatif de la campagne")
        
        # Enrichir le récapitulatif avec des informations sur le segment ciblé
        segment_info = ""
        if target_option == "Cluster" and len(target_value) == 1 and target_value[0] in cluster_recommendations:
            cluster_info = cluster_recommendations[target_value[0]]
            segment_info = f"{cluster_info['title']} - {cluster_info['strategy']}"
        
        recap = {
            "Type de campagne": campaign_type,
            "Nombre d'utilisateurs ciblés": len(filtered_df),
            "Méthode de ciblage": target_option,
            "Segment ciblé": segment_info if segment_info else ", ".join(target_value) if target_option == "Cluster" else target_value,
            "Planification": schedule_option,
            "Date d'envoi estimée": "Immédiat" if schedule_option == "Immédiatement" else f"{scheduled_date} à {scheduled_time}" if schedule_option == "Planifier pour plus tard" else f"Récurrent ({recurrence})"
        }
        
        st.json(recap)
