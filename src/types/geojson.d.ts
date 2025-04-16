declare namespace GeoJSON {
  interface Geometry {
    type: string;
    coordinates: any;
  }

  interface Feature<G = Geometry, P = any> {
    type: "Feature";
    geometry: G;
    properties?: P;
  }

  interface FeatureCollection<G = Geometry, P = any> {
    type: "FeatureCollection";
    features: Array<Feature<G, P>>;
  }
} 