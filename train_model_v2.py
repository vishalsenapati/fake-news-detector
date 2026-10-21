"""
Improved Training Script - Better Synthetic Data + Enhanced Model
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, classification_report
)
import joblib
import os
import re

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)


def create_realistic_dataset(n_samples=15000):
    """Create realistic dataset that mimics actual news patterns"""
    np.random.seed(42)
    
    # REAL NEWS: Factual, attributed, measured language, specific details
    real_patterns = [
        # Government/Policy
        "{source} reported that {government} announced a new {policy} aimed at {goal}. The {policy} will {action} {target} by {timeline}, according to {official}.",
        "The {government} has {action} a {amount} {currency} package to {goal}, {source} reported on {day}. {official} stated the measure will {impact}.",
        
        # Business/Economy
        "{company} reported {quarter} earnings of {amount} {currency}, {direction} analyst expectations. Revenue {direction} {percent}% year-over-year, driven by {driver}.",
        "The {index} {direction} {points} points to close at {level} on {day}, as {sector} stocks {direction} amid {reason}.",
        "Central bank {action} interest rates by {basis} basis points, {source} reported. The decision comes as inflation {trend} to {rate}%.",
        
        # Science/Health
        "Researchers at {institution} published a study in {journal} showing {finding}. The {study_type} involved {sample_size} participants over {duration}.",
        "Health officials {action} {guidance} regarding {topic}, citing new data from {source}. The recommendation applies to {population}.",
        "A new {treatment} for {condition} showed {result} in Phase {phase} trials, {institution} announced. The drug {mechanism}.",
        
        # Technology
        "{company} unveiled {product} featuring {feature} at {event}. The {product} will be available {timeline} starting at {price}.",
        "Researchers developed a new {technology} that can {capability}, according to a paper in {journal}. The breakthrough {implication}.",
        
        # General reporting style
        "{source} reported that {event} occurred in {location} on {day}. {official} confirmed {detail}. The incident {context}.",
        "According to {source}, {topic} has {trend} by {percent}% over {period}. Experts attribute this to {reason}.",
    ]
    
    # FAKE NEWS: Sensational, emotional, vague, conspiratorial, urgent, clickbait
    fake_patterns = [
        # Conspiracy/Cover-up
        "BREAKING: {entity} SECRETLY {action} {target} - MAINSTREAM MEDIA BLACKOUT! {source} EXPOSES the truth they don't want you to know!",
        "SHOCKING LEAK: {entity} documents reveal PLAN to {action} {target}. Share before THIS GETS DELETED!",
        "COVER-UP EXPOSED: {entity} has been {action} {target} for YEARS. Whistleblower risks everything to reveal TRUTH!",
        
        # Miracle cures/Health panic
        "MIRACLE CURE: {simple_thing} {action} {disease} OVERNIGHT! Doctors HATE this {number}-cent trick!",
        "URGENT: {common_item} found to CAUSE {scary_disease}! {entity} KNEW but HID it! Protect your family NOW!",
        "BANNED VIDEO: {entity} admits {product} {action} {harmful_effect}. Big Pharma SCRAMBLING to suppress!",
        
        # Financial scare/FOMO
        "{entity} INSIDERS dumping stock BEFORE CRASH! {indicator} signals COLLAPSE imminent. Get out NOW!",
        "SECRET {crypto} trick turns ${small} into ${large} in {short_time}! Wall Street FURIOUS this leaked!",
        "BANKS DON'T WANT YOU TO KNOW: {simple_action} eliminates debt INSTANTLY! {number} people already did it!",
        
        # Political manipulation
        "EXCLUSIVE: {politician} CAUGHT on hidden camera {action}! Election RIGGED - proof inside!",
        "{entity} PLOT to {action} {target} REVEALED! Patriot groups MOBILIZING. Time is RUNNING OUT!",
        
        # Clickbait/Emotional manipulation
        "You won't BELIEVE what {entity} did to {target}! {number} SHOCKING photos inside!",
        "This ONE WEIRD TRICK {entity} uses to {action} - EXPERTS FURIOUS it's public!",
        "LAST CHANCE: {entity} warns DISASTER unless you {action} NOW!",
        
        # Fake authority/False attribution
        "NASA CONFIRMS: {impossible_claim}! Scientists BAFFLED by {phenomenon}!",
        "CDC QUIETLY ADMITS: SHOCKING TRUTH about {common_item}! {number}% of PEOPLE affected!",
        "FBI DOCUMENTS RELEASED: {entity} {action} {target} - Media SILENT!",
    ]
    
    # Realistic fillers
    sources_real = ["Reuters", "Associated Press", "BBC News", "The Guardian", "New York Times", "Washington Post", "Wall Street Journal", "Financial Times", "Bloomberg", "CNN", "NPR", "PBS NewsHour"]
    governments = ["the Biden administration", "the White House", "Congress", "the Senate", "the House", "the Department of Health", "the EPA", "the FDA", "the CDC", "state governments"]
    companies = ["Apple", "Microsoft", "Amazon", "Google", "Meta", "Tesla", "NVIDIA", "JPMorgan Chase", "Bank of America", "Pfizer", "Moderna", "Johnson & Johnson"]
    institutions = ["Harvard University", "Stanford University", "MIT", "Johns Hopkins", "Mayo Clinic", "Cleveland Clinic", "NIH", "WHO", "CDC"]
    journals = ["Nature", "Science", "The Lancet", "NEJM", "JAMA", "Cell", "PNAS"]
    officials_real = ["a senior official", "the press secretary", "a spokesperson", "the director", "the commissioner", "the secretary"]
    
    entities_fake = ["the government", "Big Pharma", "the mainstream media", "the elites", "globalists", "the deep state", "Big Tech", "the establishment", "corporate interests"]
    simple_things = ["baking soda", "lemon water", "apple cider vinegar", "garlic", "turmeric", "coconut oil", "this common herb", "a kitchen spice"]
    diseases = ["cancer", "diabetes", "Alzheimer's", "heart disease", "COVID-19", "autism", "arthritis"]
    crypto_names = ["Bitcoin", "Ethereum", "Dogecoin", "Shiba Inu", "this altcoin", "this new token"]
    politicians = ["a senator", "a congressman", "a governor", "a cabinet member", "a federal judge"]
    
    real_texts = []
    fake_texts = []
    
    for _ in range(n_samples // 2):
        # Generate REAL news
        pattern = np.random.choice(real_patterns)
        text = pattern.format(
            source=np.random.choice(sources_real),
            government=np.random.choice(governments),
            policy=np.random.choice(["infrastructure bill", "climate initiative", "healthcare reform", "education funding", "tax reform", "trade agreement"]),
            goal=np.random.choice(["reduce emissions", "improve access", "strengthen economy", "protect consumers", "modernize systems"]),
            action=np.random.choice(["allocates", "invests", "provides", "establishes", "creates", "expands"]),
            guidance=np.random.choice(["new guidelines", "updated recommendations", "revised protocols", "safety advisories", "best practices"]),
            target=np.random.choice(["rural communities", "small businesses", "low-income families", "veterans", "students", "seniors"]),
            timeline=np.random.choice(["by 2025", "over five years", "starting next month", "by end of year", "within two years"]),
            official=np.random.choice(officials_real),
            amount=np.random.choice(["$1.2 trillion", "$500 billion", "$75 billion", "$2.3 trillion", "$120 billion"]),
            currency="",
            day=np.random.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "last week", "yesterday"]),
            impact=np.random.choice(["benefit millions", "create jobs", "reduce costs", "improve outcomes", "strengthen competitiveness"]),
            company=np.random.choice(companies),
            quarter=np.random.choice(["Q1", "Q2", "Q3", "Q4", "first quarter", "second quarter"]),
            direction=np.random.choice(["beat", "missed", "exceeded", "fell short of"]),
            percent=np.random.choice([5, 12, 3, 18, 7, 22]),
            driver=np.random.choice(["cloud growth", "advertising revenue", "strong demand", "cost cutting", "new products"]),
            index=np.random.choice(["S&P 500", "Dow Jones", "NASDAQ", "Russell 2000"]),
            points=np.random.choice([50, 120, 200, 350, 75, 300]),
            level=np.random.choice(["4,500", "35,000", "14,200", "2,100", "45,500"]),
            sector=np.random.choice(["tech", "energy", "healthcare", "financial", "consumer"]),
            reason=np.random.choice(["inflation data", "earnings reports", "Fed policy", "geopolitical tensions", "jobs report"]),
            basis=np.random.choice([25, 50, 75, 100]),
            trend=np.random.choice(["eased", "rose", "cooled", "accelerated"]),
            rate=np.random.choice(["3.2%", "4.1%", "2.8%", "3.7%"]),
            institution=np.random.choice(institutions),
            journal=np.random.choice(journals),
            finding=np.random.choice(["a link between diet and longevity", "new mechanism for drug resistance", "improved battery efficiency", "climate feedback loop", "gene therapy potential"]),
            study_type=np.random.choice(["randomized controlled trial", "meta-analysis", "longitudinal study", "observational study"]),
            sample_size=np.random.choice(["10,000", "50,000", "100,000", "25,000", "5,000"]),
            duration=np.random.choice(["five years", "ten years", "three years", "two decades"]),
            topic=np.random.choice(["vaccination schedules", "screen time guidelines", "dietary recommendations", "exercise guidelines"]),
            population=np.random.choice(["children under 12", "adults over 65", "pregnant women", "immunocompromised individuals"]),
            treatment=np.random.choice(["immunotherapy", "gene therapy", "monoclonal antibody", "antiviral", "vaccine candidate"]),
            condition=np.random.choice(["melanoma", "lung cancer", "Alzheimer's", "rare genetic disorder", "autoimmune disease"]),
            result=np.random.choice(["promising results", "significant improvement", "encouraging data", "positive safety profile"]),
            phase=np.random.choice(["2", "3"]),
            mechanism=np.random.choice(["targets specific proteins", "activates immune response", "modifies gene expression", "inhibits viral replication"]),
            product=np.random.choice(["new smartphone", "AI platform", "electric vehicle", "chip architecture", "cloud service"]),
            feature=np.random.choice(["advanced AI", "longer battery", "faster processing", "new design", "enhanced security"]),
            event=np.random.choice(["WWDC", "Google I/O", "CES", "Microsoft Build", "product launch"]),
            price=np.random.choice(["$599", "$999", "$1,299", "$799", "$399"]),
            technology=np.random.choice(["quantum computing", "fusion energy", "carbon capture", "brain-computer interface", "synthetic biology"]),
            capability=np.random.choice(["solve complex problems", "generate clean energy", "remove CO2", "restore movement", "design proteins"]),
            implication=np.random.choice(["could revolutionize computing", "may provide clean power", "could fight climate change", "offers hope for paralysis", "accelerates drug discovery"]),
            location=np.random.choice(["Washington", "New York", "California", "Texas", "Florida", "London", "Brussels", "Tokyo", "Beijing"]),
            detail=np.random.choice(["no injuries were reported", "an investigation is underway", "the situation is under control", "measures have been implemented"]),
            context=np.random.choice(["follows similar incidents last month", "comes amid heightened tensions", "is the latest in a series", "marks a significant development"]),
            period=np.random.choice(["the past year", "the last decade", "the previous quarter", "the past five years"]),
        )
        real_texts.append(text)
        
        # Generate FAKE news
        pattern = np.random.choice(fake_patterns)
        text = pattern.format(
            entity=np.random.choice(entities_fake),
            action=np.random.choice(["controls", "manipulates", "poisons", "enslaves", "brainwashes", "eliminates", "weaponizes", "tracks", "surveils"]),
            target=np.random.choice(["the population", "your children", "the food supply", "the water", "your mind", "the economy", "freedom", "democracy"]),
            source=np.random.choice(["anonymous insider", "leaked documents", "whistleblower", "suppressed report", "hidden camera"]),
            simple_thing=np.random.choice(simple_things),
            disease=np.random.choice(diseases),
            number=np.random.choice(["one", "two", "three", "five", "ten"]),
            common_item=np.random.choice(["tap water", "toothpaste", "deodorant", "microwave food", "cell phones", "WiFi", "vaccines", "fluoride"]),
            scary_disease=np.random.choice(["cancer", "infertility", "brain damage", "autism", "immune collapse"]),
            product=np.random.choice(["popular drug", "common vaccine", "widely used additive", "household chemical"]),
            harmful_effect=np.random.choice(["causes sterility", "alters DNA", "destroys immunity", "induces compliance", "triggers early death"]),
            indicator=np.random.choice(["secret metric", "insider signal", "hidden pattern", "classified algorithm"]),
            crypto=np.random.choice(crypto_names),
            small=np.random.choice(["100", "50", "200", "25"]),
            large=np.random.choice(["50,000", "100,000", "1,000,000", "250,000"]),
            short_time=np.random.choice(["48 hours", "one week", "72 hours", "30 days"]),
            simple_action=np.random.choice(["writing two words", "calling this number", "sending this letter", "filing this form"]),
            politician=np.random.choice(politicians),
            catastrophic=np.random.choice(["martial law", "currency collapse", "food shortages", "power grid failure", "civil unrest"]),
            impossible_claim=np.random.choice(["Earth is flat", "moon landing faked", "aliens live among us", "time travel is real", "gravity is a lie"]),
            phenomenon=np.random.choice(["unexplained lights", "mysterious signals", "ancient technology", "dimensional portal"]),
        )
        fake_texts.append(text)
    
    texts = real_texts + fake_texts
    labels = [0] * (n_samples // 2) + [1] * (n_samples // 2)
    
    indices = np.random.permutation(n_samples)
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    df = pd.DataFrame({'text': texts, 'label': labels})
    return df


def train_models(X, y):
    """Train and compare multiple models"""
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    models = {
        'Logistic Regression': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=15000, ngram_range=(1, 3), 
                                       stop_words='english', min_df=2, max_df=0.95,
                                       sublinear_tf=True)),
            ('clf', LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced', C=1.0))
        ]),
        'Linear SVM': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=15000, ngram_range=(1, 3),
                                       stop_words='english', min_df=2, max_df=0.95,
                                       sublinear_tf=True)),
            ('clf', LinearSVC(random_state=42, class_weight='balanced', C=1.0, max_iter=5000))
        ]),
        'Random Forest': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2),
                                       stop_words='english', min_df=3, max_df=0.95)),
            ('clf', RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced', n_jobs=-1))
        ]),
    }
    
    results = {}
    best_model = None
    best_f1 = 0
    
    for name, pipeline in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}...")
        print(f"{'='*50}")
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
        
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-Score : {f1:.4f}")
        
        print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
        
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:\n{cm}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = (name, pipeline)
    
    print(f"\n{'='*50}")
    print(f"BEST MODEL: {best_model[0]} (F1: {best_f1:.4f})")
    print(f"{'='*50}")
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [results[m][metric] for m in models.keys()]
        ax.bar(x + i*width, values, width, label=metric.capitalize(), alpha=0.8)
    
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(models.keys())
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion matrix for best model
    best_name, best_pipeline = best_model
    y_pred_best = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {best_name}', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return best_model[1], results


def save_model(pipeline, filepath='fake_news_model.joblib'):
    print(f"\nSaving model to {filepath}...")
    joblib.dump(pipeline, filepath)
    print("Model saved!")


def main():
    print("="*60)
    print("IMPROVED FAKE NEWS DETECTOR TRAINING")
    print("="*60)
    
    # Create better dataset
    print("\nGenerating realistic dataset...")
    df = create_realistic_dataset(20000)
    X, y = df['text'], df['label']
    
    print(f"Dataset: {len(X)} samples ({sum(y)} fake, {len(y)-sum(y)} real)")
    
    # Train and select best
    best_pipeline, results = train_models(X, y)
    
    # Save best model
    save_model(best_pipeline)
    
    # Save comparison results
    results_df = pd.DataFrame(results).T
    results_df.to_csv('model_comparison.csv')
    print("\nComparison saved to model_comparison.csv")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()